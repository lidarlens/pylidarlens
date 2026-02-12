"""Shared fixtures for lidarlens tests — synthetic point clouds."""

import numpy as np
import open3d as o3d
import pytest
import laspy
import os
import tempfile


@pytest.fixture
def simple_pcd():
    """A small synthetic point cloud: 1000 random points in a unit cube."""
    np.random.seed(42)
    points = np.random.rand(1000, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@pytest.fixture
def colored_pcd():
    """A small synthetic point cloud with RGB colours."""
    np.random.seed(42)
    points = np.random.rand(500, 3)
    colors = np.random.rand(500, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


@pytest.fixture
def ground_plane_pcd():
    """A synthetic point cloud with a clear ground plane + elevated points."""
    np.random.seed(42)
    n_ground = 500
    n_above = 200

    # Ground: flat plane at z ≈ 0
    ground = np.column_stack([
        np.random.uniform(0, 10, n_ground),
        np.random.uniform(0, 10, n_ground),
        np.random.normal(0, 0.05, n_ground),
    ])

    # Above ground: points at z ≈ 5
    above = np.column_stack([
        np.random.uniform(2, 8, n_above),
        np.random.uniform(2, 8, n_above),
        np.random.normal(5, 0.5, n_above),
    ])

    points = np.vstack([ground, above])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@pytest.fixture
def two_cluster_pcd():
    """Two clearly separated clusters for testing DBSCAN."""
    np.random.seed(42)
    c1 = np.random.normal(0, 0.5, (200, 3))
    c2 = np.random.normal(10, 0.5, (200, 3))
    points = np.vstack([c1, c2])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@pytest.fixture
def tmp_las_file(simple_pcd, tmp_path):
    """Write a simple LAS file for I/O testing."""
    path = str(tmp_path / "test.laz")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0, 0, 0])

    las = laspy.LasData(header)
    points = np.asarray(simple_pcd.points)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(path)
    return path


@pytest.fixture
def tmp_classified_las(tmp_path):
    """Write a LAS file with classification data."""
    path = str(tmp_path / "classified.laz")
    np.random.seed(42)
    n = 500
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0, 0, 0])

    las = laspy.LasData(header)
    las.x = np.random.rand(n) * 100
    las.y = np.random.rand(n) * 100
    las.z = np.random.rand(n) * 50
    las.classification = np.random.choice([2, 3, 5, 6], size=n).astype(np.uint8)
    las.write(path)
    return path
