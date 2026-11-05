import random

import pytest

from letree.evaluation.summary import (
    read_records,
    reduce_records,
    summarize_records,
    write_records,
)


def _record(lost, zones, pure, pure_wrong, mixed, mixed_wrong, corruption="gauss"):
    pixels = pure + mixed
    return {
        "source": "model_soft",
        "corruption": corruption,
        "groups": zones,
        "root_claims": 1,
        "valid": True,
        "failure": "",
        "exact": False,
        "refining": mixed == 0,
        "clean_zones": zones,
        "lost_zones": lost,
        "lost_area": 0,
        "root_lost": False,
        "all/pixels": pixels,
        "pure/pixels": pure,
        "mixed/pixels": mixed,
        "all/misassigned": pure_wrong + mixed_wrong,
        "pure/misassigned": pure_wrong,
        "mixed/misassigned": mixed_wrong,
        **{f"{p}/depth_error": 0 for p in ("all", "pure", "mixed")},
        "all/exact_depth": pixels,
        "pure/exact_depth": pure,
        "mixed/exact_depth": mixed,
    }


INVALID = {
    "source": "model_soft",
    "corruption": "gauss",
    "groups": 9,
    "root_claims": 4,
    "valid": False,
    "failure": "root_count",
    "exact": False,
    "refining": False,
    "clean_zones": 5,
    "all/pixels": 4096,
    "pure/pixels": 4000,
    "mixed/pixels": 96,
}

# The worked example of the 7E guide
EXAMPLE = [_record(1, 10, 4000, 40, 96, 48), _record(1, 2, 3096, 0, 1000, 100), INVALID]


def test_worked_example_reductions():
    summary = reduce_records(EXAMPLE)
    assert summary["samples"] == 3
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["exact_recovery"] == 0
    assert summary["failures/root_count"] == 1
    # Group and root claim counts include the failed decode
    assert summary["groups/sample_mean"] == pytest.approx((10 + 2 + 9) / 3)
    assert summary["root_claims/sample_mean"] == pytest.approx((1 + 1 + 4) / 3)
    assert summary["lost_clean_zones/sample_mean"] == pytest.approx(0.3)
    assert summary["lost_clean_zones/pooled"] == pytest.approx(2 / 12)
    assert summary["mixed/misassigned_pixels/sample_mean"] == pytest.approx(0.3)
    assert summary["mixed/misassigned_pixels/pooled"] == pytest.approx(148 / 1096)
    whole = (88 / 4096 + 100 / 4096) / 2
    assert summary["all/misassigned_pixels/sample_mean"] == pytest.approx(whole)
    assert summary["all/misassigned_pixels/pooled"] == pytest.approx(whole)


def test_an_invalid_record_changes_only_coverage_and_recovery():
    with_invalid = reduce_records(EXAMPLE)
    without = reduce_records(EXAMPLE[:2])
    for name, value in without.items():
        if name in (
            "samples",
            "coverage",
            "failures/root_count",
            "groups/sample_mean",
            "root_claims/sample_mean",
        ):
            continue
        assert with_invalid[name] == pytest.approx(value), name
    assert without["coverage"] == 1


def test_pixel_strata_skip_samples_without_those_pixels():
    summary = reduce_records(
        [_record(0, 4, 100, 10, 0, 0), _record(0, 4, 50, 0, 50, 25)]
    )
    assert summary["mixed/samples"] == 1
    assert summary["mixed/misassigned_pixels/sample_mean"] == pytest.approx(0.5)
    assert summary["pure/misassigned_pixels/sample_mean"] == pytest.approx(0.05)


def test_summary_does_not_depend_on_record_order():
    records = [
        _record(1, 10, 4000, 40, 96, 48),
        _record(0, 3, 4096, 0, 0, 0, corruption="clean"),
        _record(2, 7, 3000, 12, 1096, 300, corruption="speckle"),
        INVALID,
    ]
    shuffled = records[:]
    random.Random(0).shuffle(shuffled)
    assert summarize_records(records) == pytest.approx(summarize_records(shuffled))


def test_summary_groups_and_strata():
    clean = _record(0, 3, 4096, 0, 0, 0, corruption="clean")
    lost_root = _record(1, 10, 4000, 40, 96, 48) | {"root_lost": True}
    summary = summarize_records([clean, lost_root, INVALID])

    assert summary["model_soft/all/every/samples"] == 3
    assert summary["model_soft/clean/every/samples"] == 1
    assert summary["model_soft/noisy/every/samples"] == 2
    assert summary["model_soft/gauss/every/samples"] == 2
    assert summary["model_soft/all/refining/samples"] == 1
    assert summary["model_soft/all/mixed/samples"] == 2
    # Only valid outputs carry a root flag
    assert summary["model_soft/all/root_lost/samples"] == 1
    assert summary["model_soft/all/root_kept/samples"] == 1
    assert not any(key.startswith("oracle/") for key in summary)


def test_records_survive_a_csv_round_trip(tmp_path):
    path = tmp_path / "records.csv"
    write_records(path, EXAMPLE)
    assert read_records(path) == EXAMPLE
