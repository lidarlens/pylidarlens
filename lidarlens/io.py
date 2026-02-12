"""
I/O module — read and write LAS/LAZ point cloud files.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import laspy

from lidarlens.crs import DEFAULT_WKT, create_wkt_vlr


def read(
    path: str,
) -> Tuple[o3d.geometry.PointCloud, laspy.LasHeader, List, Optional[str]]:
    """Load a LAS/LAZ file and return an Open3D PointCloud plus metadata.

    Args:
        path: Path to the ``.las`` or ``.laz`` file.

    Returns:
        A tuple of ``(pcd, header, vlrs, wkt)`` where:

        - **pcd**: ``open3d.geometry.PointCloud`` with points and (optionally)
          RGB colours.
        - **header**: The original ``laspy.LasHeader``.
        - **vlrs**: List of ``laspy.VLR`` objects.
        - **wkt**: Detected WKT CRS string, or ``None``.
    """
    with laspy.open(path) as f:
        header = f.header
        vlrs = list(f.header.vlrs)
        las = f.read()

    # Extract WKT from VLRs
    wkt: Optional[str] = None
    for vlr in vlrs:
        uid = getattr(vlr, "user_id", "")
        rid = getattr(vlr, "record_id", 0)
        desc = getattr(vlr, "description", "")
        if (uid == "LASF_Projection" and rid == 2112) or "WKT" in desc:
            try:
                wkt = vlr.record_data.decode("utf-8").strip("\x00")
                break
            except Exception:
                pass

    # Build Open3D PointCloud
    points = np.vstack([las.x, las.y, las.z]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Colours
    try:
        if hasattr(las, "red"):
            red = np.array(las.red, dtype=np.float64)
            green = np.array(las.green, dtype=np.float64)
            blue = np.array(las.blue, dtype=np.float64)

            max_val = max(red.max(), green.max(), blue.max(), 1)
            scale = 65535.0 if max_val > 255 else 255.0

            colors = np.vstack([red / scale, green / scale, blue / scale]).T
            pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception:
        pass

    return pcd, header, vlrs, wkt


def save(
    pcd: o3d.geometry.PointCloud,
    output_path: str,
    *,
    header: Optional[laspy.LasHeader] = None,
    vlrs: Optional[List] = None,
    wkt: Optional[str] = None,
    extra_dims: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save an Open3D PointCloud to a LAS/LAZ file.

    Args:
        pcd: The point cloud to save.
        output_path: Destination file path (``.las`` or ``.laz``).
        header: Original LAS header to reuse offsets/scales (optional).
        vlrs: Original VLRs to preserve (optional).
        wkt: WKT CRS string to embed (optional).
        extra_dims: Mapping ``{name: numpy_array}`` of extra per-point
            dimensions (e.g. ``ClusterID``).
    """
    points = np.asarray(pcd.points)

    new_header = laspy.LasHeader(point_format=3, version="1.2")
    new_header.scales = np.array([0.001, 0.001, 0.001])
    new_header.offsets = np.min(points, axis=0)

    if header is not None:
        new_header.offsets = header.offsets
        new_header.scales = header.scales

    # Filter out COPC VLRs and add remaining
    filtered_vlrs = []
    if vlrs:
        for vlr in vlrs:
            if getattr(vlr, "user_id", "") == "copc":
                continue
            filtered_vlrs.append(vlr)

    # Check if we already have a WKT VLR
    existing_wkt = any(
        getattr(v, "user_id", "") == "LASF_Projection"
        and getattr(v, "record_id", 0) == 2112
        for v in filtered_vlrs
    )

    for vlr in filtered_vlrs:
        new_header.vlrs.append(vlr)

    # Add WKT VLR if not already present
    if not existing_wkt:
        target_wkt = wkt if wkt else DEFAULT_WKT
        new_header.vlrs.append(create_wkt_vlr(target_wkt))

    # Register extra dimensions
    if extra_dims:
        for name, data in extra_dims.items():
            dtype = np.int32 if np.issubdtype(data.dtype, np.integer) else np.float64
            new_header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))

    las = laspy.LasData(new_header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 65535).clip(0, 65535).astype(np.uint16)
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

    if extra_dims:
        for name, data in extra_dims.items():
            setattr(las, name, data)

    las.write(output_path)


def read_header(path: str) -> Dict:
    """Read only the header metadata from a LAS/LAZ file — no point data.

    Args:
        path: Path to the LAS/LAZ file.

    Returns:
        Dictionary with ``point_count``, ``point_format``, ``version``,
        ``scales``, ``offsets``, ``mins``, ``maxs``.
    """
    with laspy.open(path) as f:
        h = f.header
        return {
            "point_count": h.point_count,
            "point_format": int(h.point_format.id),
            "version": f"{h.version.major}.{h.version.minor}",
            "scales": h.scales.tolist(),
            "offsets": h.offsets.tolist(),
            "mins": h.mins.tolist(),
            "maxs": h.maxs.tolist(),
        }


def read_bounds(path: str) -> Dict:
    """Read just the spatial bounding box from a LAS/LAZ file header.

    Args:
        path: Path to the LAS/LAZ file.

    Returns:
        Dictionary with ``min_x``, ``max_x``, ``min_y``, ``max_y``,
        ``min_z``, ``max_z``.
    """
    with laspy.open(path) as f:
        h = f.header
        return {
            "min_x": float(h.mins[0]),
            "max_x": float(h.maxs[0]),
            "min_y": float(h.mins[1]),
            "max_y": float(h.maxs[1]),
            "min_z": float(h.mins[2]),
            "max_z": float(h.maxs[2]),
        }
