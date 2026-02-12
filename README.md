# LidarLens

**LidarLens** is a powerful Python package for processing, analyzing, and visualizing 3D LiDAR point cloud data. It provides a clean, pythonic API for common tasks such as I/O, noise removal, ground extraction, DTM/DSM generation, and terrain analysis.

## Features

- **I/O**: Read and write LAS/LAZ files (wraps `laspy` and `open3d`).
- **Preprocessing**: Voxel downsampling, statistical/radius outlier removal.
- **Segmentation**:
    - Ground extraction using CSF (Cloth Simulation Filter) or RANSAC.
    - Building and vegetation segmentation (heuristic).
    - DBSCAN clustering.
- **Terrain Modelling**:
    - Generate DTM (Digital Terrain Model) and DSM (Digital Surface Model).
    - Compute Canopy Height Models (CHM).
    - Generate vector contours.
- **Analysis**:
    - Point density maps, roughness, slope, aspect, and hillshade.
    - Cloud-to-cloud distance comparison.
    - Volume computation and cross-section profiling.
- **Geometry**:
    - Normal estimation and orientation.
    - Mesh surface reconstruction (Poisson, Ball Pivoting).
    - ICP Registration.
- **Visualization**: Helper functions for plotting point clouds, DEMs, and histograms.
- **CLI**: Comprehensive command-line interface for batch processing.

## Installation

```bash
pip install lidarlens
```

To install with CLI support:
```bash
pip install lidarlens[cli]
```

To install with Deep Learning support (coming soon):
```bash
pip install lidarlens[dl]
```

## Quick Start

### Python API: DEM Generation Pipeline

This example demonstrates how to import a raw point cloud, remove noise, classify ground points, and generate a Digital Terrain Model (DTM).

```python
import lidarlens as ll

# 1. Import Data
# Load your point cloud file (supports .las and .laz)
# Returns the point cloud, header, VLRs, and coordinate reference system (WKT)
pcd, header, vlrs, wkt = ll.read("raw_scan.laz")
print(f"Loaded {header.point_count} points")

# 2. Data Processing
# Remove statistical outliers (noise removal)
pcd_clean = ll.denoise(pcd, method="statistical", nb_neighbors=20, std_ratio=2.0)

# Downsample to 0.5m voxels for faster processing
pcd_down = ll.downsample(pcd_clean, voxel_size=0.5)

# 3. Ground Extraction
# Separate likely ground points from non-ground objects
ground_pcd, non_ground_pcd, plane_model = ll.extract_ground(pcd_down)

# 4. Generate DEM (Digital Terrain Model)
# Rasterize ground points into a 1.0m resolution grid
dtm, meta = ll.generate_dtm(ground_pcd, resolution=1.0)

# 5. Visualize and Save
# Save as a colorized PNG for quick visualization
ll.dem_to_png(dtm, meta, "output_dtm.png", colormap="terrain")

# Export to GeoTIFF for use in GIS software (requires rasterio)
# We pass the 'wkt' to ensure the output is properly georeferenced
ll.dem_to_geotiff(dtm, meta, "output_dtm.tif")
```

### Command Line Interface

```bash
# Info
lidarlens info input.laz

# Processing pipeline
lidarlens downsample input.laz -v 0.5 -o downsampled.laz
lidarlens denoise downsampled.laz -o clean.laz
lidarlens ground clean.laz -o ground.laz

# Terrain products
lidarlens dtm ground.laz -o dtm.png
lidarlens contours ground.laz -i 2.0 -o contours.geojson
```
