"""Tests for lidarlens.export — point cloud and DEM export."""

import numpy as np
import os
import pytest


def test_to_csv(simple_pcd, tmp_path):
    """CSV export should create a file with correct row count."""
    from lidarlens.export import to_csv

    out = str(tmp_path / "out.csv")
    actual = to_csv(simple_pcd, out)
    assert os.path.exists(actual)

    import csv
    with open(actual, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        rows = list(reader)
    assert len(rows) == len(simple_pcd.points)


def test_to_ply(simple_pcd, tmp_path):
    """PLY export should create a valid file."""
    from lidarlens.export import to_ply

    out = str(tmp_path / "out.ply")
    actual = to_ply(simple_pcd, out)
    assert os.path.exists(actual)


def test_to_numpy(simple_pcd):
    """NumPy export should return (N, 3) array."""
    from lidarlens.export import to_numpy

    arr = to_numpy(simple_pcd)
    assert arr.shape == (1000, 3)


def test_to_numpy_with_colors(colored_pcd):
    """NumPy export with colours should return (N, 6) array."""
    from lidarlens.export import to_numpy

    arr = to_numpy(colored_pcd)
    assert arr.shape[1] == 6


def test_dem_to_png(ground_plane_pcd, tmp_path):
    """DEM PNG export should create a PNG file."""
    from lidarlens.terrain import generate_dtm
    from lidarlens.export import dem_to_png

    grid, meta = generate_dtm(ground_plane_pcd, resolution=1.0)
    out = str(tmp_path / "dem.png")
    actual = dem_to_png(grid, meta, out)
    assert os.path.exists(actual)
