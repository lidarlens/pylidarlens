"""Tests for lidarlens.segmentation — ground, clustering, heuristic."""

import numpy as np
import pytest


def test_segment_plane(ground_plane_pcd):
    """RANSAC should find a dominant horizontal plane."""
    from lidarlens.segmentation import segment_plane

    model, inliers = segment_plane(ground_plane_pcd, distance_threshold=0.2)
    assert len(model) == 4
    assert len(inliers) > 0
    # The plane normal should be roughly vertical (|c| >> |a|, |b|)
    a, b, c, d = model
    assert abs(c) > abs(a) and abs(c) > abs(b)


def test_extract_ground(ground_plane_pcd):
    """Ground extraction should separate ground from non-ground."""
    from lidarlens.segmentation import extract_ground

    ground, non_ground, plane = extract_ground(
        ground_plane_pcd, distance_threshold=0.5
    )
    assert len(ground.points) > 0
    assert len(non_ground.points) > 0
    assert len(ground.points) + len(non_ground.points) == len(ground_plane_pcd.points)


def test_cluster_dbscan(two_cluster_pcd):
    """DBSCAN should find exactly 2 clusters."""
    from lidarlens.segmentation import cluster_dbscan

    labels, n = cluster_dbscan(two_cluster_pcd, eps=2.0, min_points=10)
    assert n == 2
    assert len(labels) == len(two_cluster_pcd.points)


def test_extract_clusters(two_cluster_pcd):
    """Extract clusters should return 2 separate point clouds."""
    from lidarlens.segmentation import extract_clusters

    clusters, labels = extract_clusters(
        two_cluster_pcd, eps=2.0, min_points=10, min_cluster_size=50,
    )
    assert len(clusters) == 2
    for c in clusters:
        assert len(c.points) >= 50


def test_segment_heuristic(ground_plane_pcd):
    """Heuristic segmentation should assign labels including ground (2)."""
    from lidarlens.segmentation import segment_heuristic

    labels = segment_heuristic(ground_plane_pcd)
    assert len(labels) == len(ground_plane_pcd.points)
    assert 2 in labels  # Ground class


def test_segment_csf(ground_plane_pcd):
    """CSF should separate ground from non-ground points."""
    from lidarlens.segmentation import segment_csf

    ground, non_ground = segment_csf(ground_plane_pcd)
    assert len(ground.points) > 0
    assert len(non_ground.points) > 0
