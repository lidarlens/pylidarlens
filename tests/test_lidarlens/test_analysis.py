"""Tests for lidarlens.analysis — stats, histogram, comparison, density, volume."""

import numpy as np
import pytest


def test_compute_statistics(simple_pcd):
    """Statistics should return a comprehensive dict."""
    from lidarlens.analysis import compute_statistics

    stats = compute_statistics(simple_pcd)
    assert stats["point_count"] == 1000
    assert "bounds" in stats
    assert "centroid" in stats
    assert "elevation" in stats
    assert "density" in stats


def test_elevation_histogram(simple_pcd):
    """Histogram should return counts and edges."""
    from lidarlens.analysis import elevation_histogram

    result = elevation_histogram(simple_pcd, bins=20)
    assert len(result["counts"]) == 20
    assert len(result["bin_edges"]) == 21


def test_compare_clouds(simple_pcd, colored_pcd):
    """Cloud comparison should return valid stats."""
    from lidarlens.analysis import compare_clouds

    result_pcd, distances, stats = compare_clouds(simple_pcd, colored_pcd)
    assert len(distances) == len(simple_pcd.points)
    assert stats["min_distance"] >= 0
    assert stats["rmse"] >= 0


def test_point_density(simple_pcd):
    """Point density should return a grid with positive total."""
    from lidarlens.analysis import point_density

    grid, meta = point_density(simple_pcd, resolution=0.2)
    assert grid.ndim == 2
    assert grid.sum() == len(simple_pcd.points)


def test_roughness(simple_pcd):
    """Roughness should return one value per point."""
    from lidarlens.analysis import roughness

    rough = roughness(simple_pcd, radius=0.3)
    assert len(rough) == len(simple_pcd.points)
    assert np.all(rough >= 0)


def test_compute_volume(simple_pcd):
    """Volume computation should return a positive value."""
    from lidarlens.analysis import compute_volume

    result = compute_volume(simple_pcd, resolution=0.2)
    assert result["volume"] >= 0
    assert result["area"] >= 0


def test_cross_section(ground_plane_pcd):
    """Cross-section should return distance/elevation arrays."""
    from lidarlens.analysis import cross_section

    profile = cross_section(
        ground_plane_pcd,
        line_start=[0, 5], line_end=[10, 5], width=1.0,
    )
    assert len(profile["distance"]) > 0
    assert len(profile["elevation"]) > 0
    assert profile["length"] == pytest.approx(10.0)
