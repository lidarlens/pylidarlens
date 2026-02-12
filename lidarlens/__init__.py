"""
LidarLens — A Python package for 3D LiDAR point cloud processing, analysis,
terrain modelling, and visualization.

Usage::

    import lidarlens as ll

    # Read a point cloud
    pcd, header, vlrs, wkt = ll.read("input.laz")

    # Quick statistics
    stats = ll.stats(pcd)

    # Preprocessing
    pcd_clean = ll.denoise(pcd)
    pcd_down = ll.downsample(pcd, voxel_size=0.5)

    # Segmentation
    ground, non_ground, plane = ll.extract_ground(pcd)

    # Terrain
    dtm, meta = ll.generate_dtm(ground)
    dsm, meta = ll.generate_dsm(pcd)

    # Save
    ll.save(pcd, "output.laz")
"""

from lidarlens._version import __version__

# ── I/O ──────────────────────────────────────────────────────────────────────
from lidarlens.io import read, save, read_header, read_bounds

# ── Preprocessing ────────────────────────────────────────────────────────────
from lidarlens.preprocessing import (
    voxel_downsample,
    uniform_downsample,
    random_downsample,
    denoise,
    remove_statistical_outliers,
    remove_radius_outliers,
)

# Convenience alias
downsample = voxel_downsample

# ── Filtering ────────────────────────────────────────────────────────────────
from lidarlens.filtering import (
    crop_by_bounds,
    filter_by_elevation,
    filter_by_classification,
    filter_by_return_number,
    filter_by_intensity,
    crop_by_polygon,
)

# ── Segmentation ─────────────────────────────────────────────────────────────
from lidarlens.segmentation import (
    segment_plane,
    extract_ground,
    cluster_dbscan,
    extract_clusters,
    segment_heuristic,
)

# ── Terrain ──────────────────────────────────────────────────────────────────
from lidarlens.terrain import (
    generate_dtm,
    generate_dsm,
    generate_chm,
    generate_contours,
    hillshade,
    slope_map,
    aspect_map,
)

# ── Geometry ─────────────────────────────────────────────────────────────────
from lidarlens.geometry import (
    estimate_normals,
    orient_normals,
    generate_mesh,
    translate,
    scale,
    rotate,
    register_icp,
    compute_convex_hull,
    voxelize,
)

# ── Analysis ─────────────────────────────────────────────────────────────────
from lidarlens.analysis import (
    compute_statistics,
    elevation_histogram,
    compare_clouds,
    point_density,
    roughness,
    cross_section,
    compute_volume,
)

# Convenience alias
stats = compute_statistics

# ── Classification ───────────────────────────────────────────────────────────
from lidarlens.classification import (
    ASPRS_CLASSES,
    reclassify_in_polygon,
    reclassify_by_elevation,
    get_class_distribution,
    merge_classes,
)

# ── Export ────────────────────────────────────────────────────────────────────
from lidarlens.export import (
    to_csv,
    to_ply,
    to_pcd,
    dem_to_png,
    dem_to_geotiff,
    dem_to_csv,
    to_numpy,
)

# ── CRS ──────────────────────────────────────────────────────────────────────
from lidarlens.crs import (
    create_wkt_vlr,
    transform_bounds,
    transform_coordinates,
    detect_crs,
)

__all__ = [
    "__version__",
    # I/O
    "read", "save", "read_header", "read_bounds",
    # Preprocessing
    "voxel_downsample", "uniform_downsample", "random_downsample",
    "denoise", "downsample",
    "remove_statistical_outliers", "remove_radius_outliers",
    # Filtering
    "crop_by_bounds", "filter_by_elevation", "filter_by_classification",
    "filter_by_return_number", "filter_by_intensity", "crop_by_polygon",
    # Segmentation
    "segment_plane", "extract_ground", "cluster_dbscan",
    "extract_clusters", "segment_heuristic",
    # Terrain
    "generate_dtm", "generate_dsm", "generate_chm",
    "generate_contours", "hillshade", "slope_map", "aspect_map",
    # Geometry
    "estimate_normals", "orient_normals", "generate_mesh",
    "translate", "scale", "rotate",
    "register_icp", "compute_convex_hull", "voxelize",
    # Analysis
    "compute_statistics", "stats", "elevation_histogram",
    "compare_clouds", "point_density", "roughness",
    "cross_section", "compute_volume",
    # Classification
    "ASPRS_CLASSES", "reclassify_in_polygon", "reclassify_by_elevation",
    "get_class_distribution", "merge_classes",
    # Export
    "to_csv", "to_ply", "to_pcd",
    "dem_to_png", "dem_to_geotiff", "dem_to_csv", "to_numpy",
    # CRS
    "create_wkt_vlr", "transform_bounds", "transform_coordinates", "detect_crs",
]
