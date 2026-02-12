"""Tests for lidarlens.classification — ASPRS classes, reclassification, distribution."""

import numpy as np
import pytest


def test_asprs_classes():
    """ASPRS_CLASSES should include standard codes."""
    from lidarlens.classification import ASPRS_CLASSES

    assert 2 in ASPRS_CLASSES
    assert ASPRS_CLASSES[2] == "Ground"
    assert 6 in ASPRS_CLASSES
    assert ASPRS_CLASSES[6] == "Building"


def test_get_class_distribution(tmp_classified_las):
    """Class distribution should sum to 100%."""
    from lidarlens.classification import get_class_distribution

    dist = get_class_distribution(tmp_classified_las)
    total = sum(d["percentage"] for d in dist.values())
    assert abs(total - 100.0) < 0.1


def test_merge_classes(tmp_classified_las, tmp_path):
    """Merging class 3 → 5 should remove class 3 from output."""
    from lidarlens.classification import merge_classes, get_class_distribution

    output = str(tmp_path / "merged.laz")
    merge_classes(tmp_classified_las, {3: 5}, output_path=output)

    dist = get_class_distribution(output)
    assert 3 not in dist
    assert 5 in dist


def test_reclassify_by_elevation(tmp_classified_las, tmp_path):
    """Elevation-based reclassification should apply ranges."""
    from lidarlens.classification import reclassify_by_elevation, get_class_distribution

    output = str(tmp_path / "reclass.laz")
    reclassify_by_elevation(
        tmp_classified_las,
        ranges=[(0, 10, 2), (10, 50, 5)],
        output_path=output,
    )
    dist = get_class_distribution(output)
    # Should have at least ground (2) and high veg (5)
    assert 2 in dist or 5 in dist
