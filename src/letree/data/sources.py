"""Clean image sources, written as one grayscale PNG per image and split.

A source folder holds `source.json` and one subfolder per split. Families are
built from any such folder, so a new image collection only needs its own
writer here.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from letree.generation.images import SquareImageConfig, generate_square_image
from letree.generation.seeds import GenerationStage, derive_rng


def write_square_source(
    root: str | Path,
    base_seed: int,
    split_image_counts: dict[str, int],
    config: SquareImageConfig,
) -> None:
    root = Path(root)
    _start_source(
        root,
        {
            "source": "square_images",
            "base_seed": base_seed,
            "image_config": asdict(config),
            "split_image_counts": split_image_counts,
        },
    )
    # Splits take consecutive index ranges so each image has its own stream
    index = 0
    for split, count in split_image_counts.items():
        (root / split).mkdir()
        for _ in range(count):
            rng = derive_rng(base_seed, index, GenerationStage.SQUARE_IMAGE)
            image = generate_square_image(config, rng)
            Image.fromarray(image).save(root / split / f"square_{index:08d}.png")
            index += 1


def _start_source(root: Path, description: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source.json"
    if path.exists():
        raise FileExistsError(f"A source already exists in {root}.")
    path.write_text(json.dumps(description, indent=2))


def source_images(root: str | Path) -> list[tuple[str, Path]]:
    """Every image as (split, path), splits in source order and names sorted."""
    root = Path(root)
    description = json.loads((root / "source.json").read_text())
    return [
        (split, path)
        for split in description["split_image_counts"]
        for path in sorted((root / split).glob("*.png"))
    ]


def read_image(path: Path) -> NDArray[np.uint8]:
    image = np.asarray(Image.open(path))
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError(f"{path} is not an 8-bit grayscale image.")
    return image
