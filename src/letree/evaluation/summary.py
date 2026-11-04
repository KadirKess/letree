"""Reductions of score records per source, corruption group and stratum."""

import csv
from pathlib import Path

import numpy as np

from letree.correction.decoder import DecodeFailureReason
from letree.evaluation.records import SOURCES

PIXEL_SETS = ("all", "pure", "mixed")


def reduce_records(records: list[dict]) -> dict[str, float]:
    """Coverage, exact recovery, and scores over the valid decodes."""
    valid = [record for record in records if record["valid"]]
    summary = {
        "samples": len(records),
        "coverage": len(valid) / len(records),
        "exact_recovery": float(np.mean([record["exact"] for record in records])),
    }
    for reason in DecodeFailureReason:
        summary[f"failures/{reason}"] = sum(
            record["failure"] == reason for record in records
        )
    if not valid:
        return summary

    def reduce(name, numerators, denominators):
        numerators, denominators = np.asarray(numerators), np.asarray(denominators)
        summary[f"{name}/sample_mean"] = float(np.mean(numerators / denominators))
        summary[f"{name}/pooled"] = float(numerators.sum() / denominators.sum())

    reduce(
        "lost_clean_zones",
        [record["lost_zones"] for record in valid],
        [record["clean_zones"] for record in valid],
    )
    reduce(
        "lost_clean_area",
        [record["lost_area"] for record in valid],
        [record["all/pixels"] for record in valid],
    )
    summary["clean_root_lost"] = float(
        np.mean([record["root_lost"] for record in valid])
    )

    for pixels in PIXEL_SETS:
        # A sample without pixels of this kind has no fraction to average
        holding = [record for record in valid if record[f"{pixels}/pixels"] > 0]
        summary[f"{pixels}/samples"] = len(holding)
        if not holding:
            continue
        counts = [record[f"{pixels}/pixels"] for record in holding]
        for name, field in (
            ("misassigned_pixels", "misassigned"),
            ("depth_mae", "depth_error"),
            ("exact_depth", "exact_depth"),
        ):
            reduce(
                f"{pixels}/{name}",
                [record[f"{pixels}/{field}"] for record in holding],
                counts,
            )
    return summary


def summarize_records(records: list[dict]) -> dict[str, float]:
    """Reductions keyed source/group/stratum/quantity."""
    noise_kinds = sorted({record["corruption"] for record in records} - {"clean"})
    groups = {
        "all": lambda record: True,
        "clean": lambda record: record["corruption"] == "clean",
        "noisy": lambda record: record["corruption"] != "clean",
        **{
            kind: lambda record, kind=kind: record["corruption"] == kind
            for kind in noise_kinds
        },
    }
    strata = {
        "every": lambda record: True,
        "refining": lambda record: record["refining"],
        "mixed": lambda record: not record["refining"],
        "root_lost": lambda record: record["valid"] and record["root_lost"],
        "root_kept": lambda record: record["valid"] and not record["root_lost"],
    }
    summary = {}
    for source in SOURCES:
        for group, in_group in groups.items():
            for stratum, in_stratum in strata.items():
                chosen = [
                    record
                    for record in records
                    if record["source"] == source
                    and in_group(record)
                    and in_stratum(record)
                ]
                if not chosen:
                    continue
                for name, value in reduce_records(chosen).items():
                    summary[f"{source}/{group}/{stratum}/{name}"] = value
    return summary


def write_records(path: Path, records: list[dict]) -> None:
    # Invalid records lack the score fields, so the header is the union of keys
    fields = list(dict.fromkeys(name for record in records for name in record))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(records)


def read_records(path: Path) -> list[dict]:

    def parse(value: str):
        if value in ("True", "False"):
            return value == "True"
        if value.lstrip("-").isdigit():
            return int(value)
        return value

    with path.open(newline="") as file:
        return [
            # Score fields of invalid records come back empty
            {name: parse(value) for name, value in row.items() if value != ""}
            | {"failure": row["failure"]}
            for row in csv.DictReader(file)
        ]
