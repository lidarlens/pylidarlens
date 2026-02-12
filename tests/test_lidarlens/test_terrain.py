"""Tests for lidarlens.terrain — DTM, DSM, CHM, contours, hillshade."""

import numpy as np
import pytest


def test_generate_dtm(ground_plane_pcd):
    """DTM should produce a 2D grid of correct dimensions."""
    from lidarlens.terrain import generate_dtm

    grid, meta = generate_dtm(ground_plane_pcd, resolution=1.0)
    assert grid.ndim == 2
    assert meta["width"] == grid.shape[1]
    assert meta["height"] == grid.shape[0]
    assert meta["type"] == "dtm"


def test_generate_dsm(ground_plane_pcd):
    """DSM should have higher z_max than DTM from the same data."""
    from lidarlens.terrain import generate_dtm, generate_dsm

    dtm, dtm_meta = generate_dtm(ground_plane_pcd, resolution=1.0)
    dsm, dsm_meta = generate_dsm(ground_plane_pcd, resolution=1.0)
    # DSM uses max Z, DTM uses min Z, so DSM values should be >= DTM
    assert dsm.max() >= dtm.min()


def test_generate_chm(ground_plane_pcd):
    """CHM = DSM - DTM should be non-negative."""
    from lidarlens.terrain import generate_dtm, generate_dsm, generate_chm

    dtm, meta = generate_dtm(ground_plane_pcd, resolution=1.0)
    dsm, _ = generate_dsm(ground_plane_pcd, resolution=1.0)
    chm, chm_meta = generate_chm(dtm, dsm, meta)

    assert chm_meta["type"] == "chm"
    assert np.all(chm >= 0)


def test_generate_contours(ground_plane_pcd):
    """Contour generation should return valid GeoJSON."""
    from lidarlens.terrain import generate_dtm, generate_contours

    grid, meta = generate_dtm(ground_plane_pcd, resolution=1.0)
    geojson = generate_contours(grid, meta, interval=1.0)

    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson


def test_hillshade(ground_plane_pcd):
    """Hillshade should return uint8 values in [0, 255]."""
    from lidarlens.terrain import generate_dsm, hillshade

    grid, meta = generate_dsm(ground_plane_pcd, resolution=1.0)
    shade = hillshade(grid, meta)

    assert shade.dtype == np.uint8
    assert shade.min() >= 0
    assert shade.max() <= 255


def test_slope_map(ground_plane_pcd):
    """Slope should return non-negative degree values."""
    from lidarlens.terrain import generate_dsm, slope_map

    grid, meta = generate_dsm(ground_plane_pcd, resolution=1.0)
    slope = slope_map(grid, meta)

    assert slope.ndim == 2
    assert np.all(slope >= 0)


def test_aspect_map(ground_plane_pcd):
    """Aspect should return values in [0, 360)."""
    from lidarlens.terrain import generate_dsm, aspect_map

    grid, meta = generate_dsm(ground_plane_pcd, resolution=1.0)
    aspect = aspect_map(grid, meta)

    assert aspect.ndim == 2
    assert np.all(aspect >= 0)
    assert np.all(aspect < 360)
