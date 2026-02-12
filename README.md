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

### Python API

```python
import lidarlens as ll

# 1. Load a point cloud
pcd, header, vlrs, wkt = ll.read("input.laz")

# 2. Preprocess
pcd = ll.denoise(pcd, method="statistical")
pcd = ll.downsample(pcd, voxel_size=0.5)

# 3. Extract Ground
ground, non_ground, plane = ll.extract_ground(pcd)

# 4. Generate DTM
dtm, meta = ll.generate_dtm(ground, resolution=1.0)

# 5. Save results
ll.save(ground, "ground.laz", header=header, wkt=wkt)
ll.dem_to_png(dtm, meta, "dtm.png")
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
