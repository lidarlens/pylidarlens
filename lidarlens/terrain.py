"""
Terrain module — DTM/DSM/CHM generation, contour extraction, and terrain
analysis (hillshade, slope, aspect).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

try:
    import scipy.ndimage
except ImportError:
    scipy = None  # type: ignore[assignment]

try:
    import pyproj
except ImportError:
    pyproj = None

from lidarlens.crs import transform_bounds


# ── DEM Generation ───────────────────────────────────────────────────────────


def _build_grid(
    pcd: o3d.geometry.PointCloud,
    resolution: float,
    aggregation: str = "min",
    smooth_sigma: float = 1.0,
    wkt: Optional[str] = None,
    dem_type: str = "dtm",
) -> Tuple[np.ndarray, Dict]:
    """Internal helper to rasterise a point cloud onto a regular grid."""
    points = np.asarray(pcd.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    z_min, z_max = float(z.min()), float(z.max())

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution)))

    init_val = np.inf if aggregation == "min" else -np.inf
    grid = np.full((height, width), init_val)

    idx_x = np.clip(((x - x_min) / resolution).astype(int), 0, width - 1)
    idx_y = np.clip(((y - y_min) / resolution).astype(int), 0, height - 1)

    if aggregation == "min":
        for i in range(len(z)):
            r, c = idx_y[i], idx_x[i]
            if z[i] < grid[r, c]:
                grid[r, c] = z[i]
        grid[grid == np.inf] = z_min
    else:  # max
        for i in range(len(z)):
            r, c = idx_y[i], idx_x[i]
            if z[i] > grid[r, c]:
                grid[r, c] = z[i]
        grid[grid == -np.inf] = z_min

    if smooth_sigma > 0 and scipy is not None:
        grid = scipy.ndimage.gaussian_filter(grid, sigma=smooth_sigma)

    metadata: Dict = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "resolution": resolution,
        "width": width, "height": height,
        "type": dem_type,
    }

    # Add WGS84 bounds if CRS available
    if wkt:
        try:
            bounds = {
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
            }
            metadata["bounds_wgs84"] = transform_bounds(bounds, wkt)
        except Exception:
            pass

    return grid, metadata


def generate_dtm(
    pcd: o3d.geometry.PointCloud,
    resolution: float = 1.0,
    smooth_sigma: float = 1.0,
    wkt: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate a Digital Terrain Model (DTM) using minimum elevation per cell.

    Args:
        pcd: Input point cloud (ideally ground-only points).
        resolution: Grid cell size in the point cloud's spatial units.
        smooth_sigma: Gaussian smoothing sigma. Set to 0 for no smoothing.
        wkt: CRS as WKT or EPSG code string (e.g. ``"EPSG:2193"``).

    Returns:
        ``(grid, metadata)`` — the 2-D elevation array and a metadata dict.
    """
    return _build_grid(pcd, resolution, "min", smooth_sigma, wkt, "dtm")


def generate_dsm(
    pcd: o3d.geometry.PointCloud,
    resolution: float = 1.0,
    smooth_sigma: float = 1.0,
    wkt: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate a Digital Surface Model (DSM) using maximum elevation per cell.

    Args:
        pcd: Input point cloud (all returns).
        resolution: Grid cell size.
        smooth_sigma: Gaussian smoothing sigma.
        wkt: CRS string.

    Returns:
        ``(grid, metadata)``
    """
    return _build_grid(pcd, resolution, "max", smooth_sigma, wkt, "dsm")


def generate_chm(
    dtm: np.ndarray,
    dsm: np.ndarray,
    metadata: Dict,
) -> Tuple[np.ndarray, Dict]:
    """Generate a Canopy Height Model (CHM) as DSM − DTM.

    Args:
        dtm: 2-D array from :func:`generate_dtm`.
        dsm: 2-D array from :func:`generate_dsm`.
        metadata: Metadata from either DTM or DSM generation.

    Returns:
        ``(chm_array, metadata)`` — negative values are clipped to 0.
    """
    chm = np.maximum(dsm - dtm, 0.0)
    meta = dict(metadata)
    meta["type"] = "chm"
    meta["z_min"] = float(chm.min())
    meta["z_max"] = float(chm.max())
    return chm, meta


# ── Contour Generation ───────────────────────────────────────────────────────


def _simplify_points(points: List, tolerance: float) -> List:
    """Distance-based vertex simplification."""
    if len(points) < 3:
        return points
    simplified = [points[0]]
    last_x, last_y = points[0]
    tol_sq = tolerance * tolerance
    for i in range(1, len(points)):
        x, y = points[i]
        dx, dy = x - last_x, y - last_y
        if dx * dx + dy * dy > tol_sq:
            simplified.append(points[i])
            last_x, last_y = x, y
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified


def generate_contours(
    dem_array: np.ndarray,
    metadata: Dict,
    interval: float = 5.0,
    smooth: bool = True,
    wkt: Optional[str] = None,
) -> Dict:
    """Generate contour lines from a DEM array as GeoJSON.

    Args:
        dem_array: 2-D elevation array from :func:`generate_dtm` or
            :func:`generate_dsm`.
        metadata: Metadata dict from DTM/DSM generation.
        interval: Contour interval in elevation units.
        smooth: Whether to smooth contour lines (placeholder — currently
            contour lines come from matplotlib which applies some smoothing).
        wkt: CRS for reprojection to EPSG:4326.

    Returns:
        GeoJSON ``FeatureCollection`` dictionary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_min = metadata["z_min"]
    z_max = metadata["z_max"]
    x_min = metadata["x_min"]
    y_min = metadata["y_min"]
    resolution = metadata["resolution"]

    levels = np.arange(
        np.floor(z_min / interval) * interval,
        np.ceil(z_max / interval) * interval + interval,
        interval,
    )

    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    cs = ax.contour(dem_array, levels=levels)
    plt.close(fig)

    # Build optional CRS transformer
    transformer = None
    if wkt and pyproj is not None:
        try:
            crs_src = pyproj.CRS.from_user_input(wkt)
            if not crs_src.is_geographic:
                crs_dst = pyproj.CRS.from_epsg(4326)
                transformer = pyproj.Transformer.from_crs(
                    crs_src, crs_dst, always_xy=True,
                )
        except Exception:
            pass

    features = []
    for level_idx, level_val in enumerate(cs.levels):
        segs = cs.allsegs[level_idx] if level_idx < len(cs.allsegs) else []
        for seg in segs:
            if len(seg) < 2:
                continue

            orig_coords = []
            for pt in seg:
                wx = x_min + pt[0] * resolution
                wy = y_min + pt[1] * resolution
                if math.isfinite(wx) and math.isfinite(wy):
                    orig_coords.append([wx, wy])
            if len(orig_coords) < 2:
                continue

            tol = max(0.1, resolution * 0.2)
            simplified = _simplify_points(orig_coords, tol)

            final_coords = []
            for x, y in simplified:
                tx, ty = x, y
                if transformer:
                    try:
                        tx, ty = transformer.transform(x, y)
                    except Exception:
                        pass
                if math.isfinite(tx) and math.isfinite(ty):
                    final_coords.append([round(float(tx), 7), round(float(ty), 7)])

            if len(final_coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": final_coords,
                    },
                    "properties": {"elevation": float(level_val)},
                })

    return {"type": "FeatureCollection", "features": features}


# ── Terrain Analysis ─────────────────────────────────────────────────────────


def hillshade(
    dem_array: np.ndarray,
    metadata: Dict,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """Compute a hillshade rendering from a DEM.

    Args:
        dem_array: 2-D elevation array.
        metadata: Metadata dict with ``resolution``.
        azimuth: Light source azimuth in degrees (0 = north, 90 = east).
        altitude: Light source altitude in degrees above horizon.

    Returns:
        2-D array of hillshade values in ``[0, 255]``.
    """
    res = metadata["resolution"]
    az_rad = np.radians(360.0 - azimuth + 90.0)
    alt_rad = np.radians(altitude)

    dy, dx = np.gradient(dem_array, res)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dy, dx)

    shade = (
        np.sin(alt_rad) * np.cos(slope)
        + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    shade = np.clip(shade, 0, 1) * 255
    return shade.astype(np.uint8)


def slope_map(
    dem_array: np.ndarray,
    metadata: Dict,
) -> np.ndarray:
    """Compute the slope angle (in degrees) for each DEM cell.

    Args:
        dem_array: 2-D elevation array.
        metadata: Metadata dict with ``resolution``.

    Returns:
        2-D array of slope angles in degrees.
    """
    res = metadata["resolution"]
    dy, dx = np.gradient(dem_array, res)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))


def aspect_map(
    dem_array: np.ndarray,
    metadata: Dict,
) -> np.ndarray:
    """Compute the aspect (compass direction of steepest descent) for each
    DEM cell.

    Args:
        dem_array: 2-D elevation array.
        metadata: Metadata dict with ``resolution``.

    Returns:
        2-D array of aspect values in degrees (0–360, north = 0).
    """
    res = metadata["resolution"]
    dy, dx = np.gradient(dem_array, res)
    aspect = np.degrees(np.arctan2(-dy, dx))
    # Convert from math angles to compass bearing
    aspect = (90.0 - aspect) % 360.0
    return aspect
