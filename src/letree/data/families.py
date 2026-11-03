"""Families of clean and noisy observations, generated and stored one file each."""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from letree.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from letree.data.sources import read_image, source_images
from letree.generation.noise import (
    NOISE_KIND_NAMES,
    AdditiveGaussianNoise,
    NoiseConfig,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
)
from letree.generation.samples import TreeImage, build_generation_sample
from letree.generation.seeds import GenerationStage, derive_rng
from letree.generation.targets import (
    CorrectionAction,
    CorrectionDiagnostics,
    CorrectionTargets,
)

CORRECTION_TARGET_POLICY = "pixel_count_representative_v1"

# Each corruption family draws from its own stream so that adding one family
# never changes the realizations of another.
NOISE_STAGES = {
    SaltAndPepperNoise: GenerationStage.SALT_AND_PEPPER_NOISE,
    AdditiveGaussianNoise: GenerationStage.ADDITIVE_GAUSSIAN_NOISE,
    SpeckleNoise: GenerationStage.SPECKLE_NOISE,
    PoissonNoise: GenerationStage.POISSON_NOISE,
}


@dataclass
class Observation:
    """One observed tree image of a family and its correction supervision."""

    noise: str
    realization: int
    tree_image: TreeImage
    targets: CorrectionTargets
    diagnostics: CorrectionDiagnostics


def build_family(
    clean_image: NDArray[np.uint8],
    base_seed: int,
    sample_index: int,
    kind: ComponentTreeKind,
    noise_configs: tuple[NoiseConfig, ...],
    noisy_count: int,
) -> list[Observation]:
    """The clean observation, then noisy_count observations with kinds in rotation.

    Observation j uses kind (sample_index + j) mod k among the k noise kinds, so
    consecutive images start one kind later and every kind gets the same share
    of a split. A kind drawn again on one image gets the next realization.
    """
    # Two configurations of one kind would read the same stream
    kinds = [type(noise_config) for noise_config in noise_configs]
    if len(set(kinds)) != len(kinds):
        raise ValueError("Each noise kind may appear only once in a family.")

    clean = build_generation_sample(clean_image, kind)
    family = [
        Observation(
            "clean",
            0,
            clean.observed,
            clean.correction_targets,
            clean.correction_diagnostics,
        )
    ]
    for j in range(noisy_count):
        noise_config = noise_configs[(sample_index + j) % len(noise_configs)]
        realization = j // len(noise_configs)
        rng = derive_rng(
            base_seed,
            sample_index,
            NOISE_STAGES[type(noise_config)],
            realization_index=realization,
        )
        sample = build_generation_sample(clean_image, kind, noise_config, rng)
        family.append(
            Observation(
                NOISE_KIND_NAMES[type(noise_config)],
                realization,
                sample.observed,
                sample.correction_targets,
                sample.correction_diagnostics,
            )
        )
    return family


def check_family(family: list[Observation]) -> None:
    """Checks run once before a family is written, never on loading."""
    keys = [(observation.noise, observation.realization) for observation in family]
    if len(set(keys)) != len(keys):
        raise ValueError("Realization indices must be unique within a noise kind.")
    for observation in family:
        check_targets(observation.targets)


def check_targets(targets: CorrectionTargets) -> None:
    """Corrective parents must form one tree under a self-parented KEEP_ZERO root."""
    parents, actions = targets.corrective_parents, targets.actions
    node_count = parents.size
    if np.any((parents < 0) | (parents >= node_count)):
        raise ValueError("corrective_parents must use local observed node identifiers.")
    roots = np.flatnonzero(parents == np.arange(node_count))
    if roots.size != 1 or actions[roots[0]] != CorrectionAction.KEEP_ZERO:
        raise ValueError(
            "Correction targets must have one self-parented KEEP_ZERO root."
        )

    # A node is settled once its path reaches the root. Meeting a node of the
    # current path again before that means a cycle.
    settled = np.zeros(node_count, dtype=bool)
    settled[roots[0]] = True
    for start in range(node_count):
        path, node = set(), start
        while not settled[node]:
            if node in path:
                raise ValueError(
                    "Correction pointers contain a cycle outside the root."
                )
            path.add(node)
            node = int(parents[node])
        settled[list(path)] = True


TREE_FIELDS = ("label_map", "edge_index", "levels", "pixel_counts", "anchors")
DIAGNOSTIC_FIELDS = (
    "overlap_clean_zone_count",
    "overlap_purity",
    "missing_clean_zones",
    "clean_root_missing",
    "artificial_root_attachment_count",
)


def save_family(path: Path, family: list[Observation], source: str) -> None:
    """Write a family with the name of the source image it was built from."""
    check_family(family)
    arrays = {
        "source": np.asarray(source),
        "kind": np.asarray(family[0].tree_image.component_tree.kind.value),
        "noise": np.asarray([observation.noise for observation in family]),
        "realization": np.asarray([observation.realization for observation in family]),
    }
    for position, observation in enumerate(family):
        tree_image = observation.tree_image
        arrays[f"{position}/image"] = tree_image.image
        arrays[f"{position}/parents"] = tree_image.component_tree.parents
        for name in TREE_FIELDS:
            arrays[f"{position}/{name}"] = getattr(tree_image.graph, name)
        arrays[f"{position}/corrective_parents"] = (
            observation.targets.corrective_parents
        )
        arrays[f"{position}/actions"] = observation.targets.actions
        for name in DIAGNOSTIC_FIELDS:
            arrays[f"{position}/{name}"] = np.asarray(
                getattr(observation.diagnostics, name)
            )

    # Write then rename, so an interrupted run never leaves a truncated family
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def load_observation(path: Path, position: int) -> Observation:
    """One observation, read without decoding the rest of its family."""
    with np.load(path) as arrays:
        return _read_observation(arrays, position)


def load_family(path: Path) -> list[Observation]:
    with np.load(path) as arrays:
        return [_read_observation(arrays, i) for i in range(arrays["noise"].size)]


def _read_observation(arrays, position: int) -> Observation:
    def read(name):
        return arrays[f"{position}/{name}"]

    tree_image = TreeImage(
        image=read("image"),
        graph=FlatZoneGraph(**{name: read(name) for name in TREE_FIELDS}),
        component_tree=ComponentTree(
            ComponentTreeKind(str(arrays["kind"])), read("parents")
        ),
    )
    diagnostics = {name: read(name) for name in DIAGNOSTIC_FIELDS}
    diagnostics["clean_root_missing"] = bool(diagnostics["clean_root_missing"])
    diagnostics["artificial_root_attachment_count"] = int(
        diagnostics["artificial_root_attachment_count"]
    )
    return Observation(
        noise=str(arrays["noise"][position]),
        realization=int(arrays["realization"][position]),
        tree_image=tree_image,
        targets=CorrectionTargets(read("corrective_parents"), read("actions")),
        diagnostics=CorrectionDiagnostics(**diagnostics),
    )


def family_path(root: Path, split: str, sample_index: int) -> Path:
    return Path(root) / split / f"sample_{sample_index:08d}.npz"


def generate_dataset(
    root: str | Path,
    source: str | Path,
    base_seed: int,
    kind: ComponentTreeKind,
    noise_configs: tuple[NoiseConfig, ...],
    noisy_counts: dict[str, int],
    workers: int = 1,
) -> None:
    """Build one family per clean image of a source folder, keeping its splits.

    noisy_counts gives the number of noisy observations per image in each split.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"A dataset already exists in {root}.")

    # An image's position in the source fixes its noise streams, so the same
    # source and seed always give the same families
    images = source_images(source)
    missing = {split for split, _ in images} - set(noisy_counts)
    if missing:
        raise ValueError(f"No noisy count given for splits {sorted(missing)}.")
    jobs = [
        (split, index, path, noisy_counts[split])
        for index, (split, path) in enumerate(images)
    ]
    split_counts = {}
    for split, _ in images:
        split_counts[split] = split_counts.get(split, 0) + 1
        (root / split).mkdir(exist_ok=True)

    manifest = {
        "base_seed": base_seed,
        "component_tree_kind": kind.value,
        "connectivity": 4,
        "correction_target_policy": CORRECTION_TARGET_POLICY,
        "noise_configs": {
            NOISE_KIND_NAMES[type(config)]: asdict(config) for config in noise_configs
        },
        "noisy_counts": noisy_counts,
        "split_sample_counts": split_counts,
        "source": json.loads((Path(source) / "source.json").read_text()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Families have their own streams and files, so workers can write them in any order
    write = partial(_write_family, root, base_seed, kind, noise_configs)
    progress = tqdm(total=len(jobs), desc="families", mininterval=5)
    if workers == 1:
        for job in jobs:
            write(job)
            progress.update()
    else:
        with ProcessPoolExecutor(workers) as pool:
            for _ in pool.map(write, jobs, chunksize=32):
                progress.update()
    progress.close()


def _write_family(root, base_seed, kind, noise_configs, job) -> None:
    split, sample_index, path, noisy_count = job
    family = build_family(
        read_image(path), base_seed, sample_index, kind, noise_configs, noisy_count
    )
    save_family(family_path(root, split, sample_index), family, path.name)
