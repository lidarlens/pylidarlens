"""
CLI module — Click-based command-line interface for common LiDAR workflows.

Usage::

    lidarlens info input.laz
    lidarlens downsample input.laz -v 0.5 -o downsampled.laz
    lidarlens denoise input.laz -m statistical -o clean.laz
    lidarlens ground input.laz -o ground.laz
    lidarlens dtm input.laz -r 1.0 -o dem.png
    lidarlens contours input.laz -i 5.0 -o contours.geojson
    lidarlens export input.laz -f ply -o output.ply
    lidarlens compare a.laz b.laz
"""

from __future__ import annotations

import json
import sys

try:
    import click
except ImportError:
    # Provide a clear error when click is not installed
    def main():
        print(
            "The 'click' package is required for the CLI. "
            "Install it with: pip install lidarlens[cli]",
            file=sys.stderr,
        )
        sys.exit(1)
else:
    @click.group()
    @click.version_option(package_name="lidarlens")
    def main():
        """LidarLens — 3D LiDAR point cloud processing toolkit."""
        pass

    @main.command()
    @click.argument("input_file")
    def info(input_file: str):
        """Print point cloud statistics."""
        from lidarlens.io import read
        from lidarlens.analysis import compute_statistics

        click.echo(f"Loading {input_file}...")
        pcd, header, vlrs, wkt = read(input_file)
        stats = compute_statistics(pcd)

        click.echo(f"\n{'─' * 50}")
        click.echo(f"  File: {input_file}")
        click.echo(f"  CRS:  {wkt[:60] + '...' if wkt and len(wkt) > 60 else wkt or 'Unknown'}")
        click.echo(f"{'─' * 50}")
        click.echo(f"  Points:    {stats['point_count']:,}")
        click.echo(f"  Bounds X:  [{stats['bounds']['min_x']:.2f}, {stats['bounds']['max_x']:.2f}]")
        click.echo(f"  Bounds Y:  [{stats['bounds']['min_y']:.2f}, {stats['bounds']['max_y']:.2f}]")
        click.echo(f"  Bounds Z:  [{stats['bounds']['min_z']:.2f}, {stats['bounds']['max_z']:.2f}]")
        click.echo(f"  Centroid:  ({stats['centroid']['x']:.2f}, {stats['centroid']['y']:.2f}, {stats['centroid']['z']:.2f})")
        click.echo(f"  Area:      {stats['density']['area']:.1f}")
        click.echo(f"  Density:   {stats['density']['points_per_unit']:.1f} pts/unit²")
        click.echo(f"  Colors:    {'Yes' if stats['has_colors'] else 'No'}")
        click.echo(f"  Normals:   {'Yes' if stats['has_normals'] else 'No'}")
        click.echo(f"{'─' * 50}")

    @main.command()
    @click.argument("input_file")
    @click.option("-v", "--voxel-size", default=0.5, help="Voxel size for downsampling.")
    @click.option("-o", "--output", default=None, help="Output file path.")
    def downsample(input_file: str, voxel_size: float, output: str | None):
        """Voxel-downsample a point cloud."""
        from lidarlens.io import read, save
        from lidarlens.preprocessing import voxel_downsample

        click.echo(f"Loading {input_file}...")
        pcd, header, vlrs, wkt = read(input_file)
        n_before = len(pcd.points)

        click.echo(f"Downsampling with voxel_size={voxel_size}...")
        pcd_out = voxel_downsample(pcd, voxel_size)
        n_after = len(pcd_out.points)

        if output is None:
            output = input_file.replace(".", f"_downsampled_v{voxel_size}.")

        save(pcd_out, output, header=header, vlrs=vlrs, wkt=wkt)
        click.echo(f"Done: {n_before:,} → {n_after:,} points ({n_after/n_before*100:.1f}%)")
        click.echo(f"Saved to: {output}")

    @main.command()
    @click.argument("input_file")
    @click.option("-m", "--method", default="statistical", type=click.Choice(["statistical", "radius"]))
    @click.option("--nb-neighbors", default=20, help="Number of neighbours (statistical).")
    @click.option("--std-ratio", default=2.0, help="Std-dev ratio (statistical).")
    @click.option("-o", "--output", default=None, help="Output file path.")
    def denoise(input_file: str, method: str, nb_neighbors: int, std_ratio: float, output: str | None):
        """Remove noise from a point cloud."""
        from lidarlens.io import read, save
        from lidarlens.preprocessing import denoise as _denoise

        click.echo(f"Loading {input_file}...")
        pcd, header, vlrs, wkt = read(input_file)
        n_before = len(pcd.points)

        click.echo(f"Denoising with method={method}...")
        pcd_out = _denoise(pcd, method=method, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        n_after = len(pcd_out.points)

        if output is None:
            output = input_file.replace(".", f"_denoised.")

        save(pcd_out, output, header=header, vlrs=vlrs, wkt=wkt)
        click.echo(f"Done: {n_before:,} → {n_after:,} points (removed {n_before-n_after:,} outliers)")
        click.echo(f"Saved to: {output}")

    @main.command()
    @click.argument("input_file")
    @click.option("-o", "--output", default=None, help="Output file path.")
    @click.option("-d", "--distance-threshold", default=0.3, help="RANSAC distance threshold.")
    def ground(input_file: str, output: str | None, distance_threshold: float):
        """Extract ground points."""
        from lidarlens.io import read, save
        from lidarlens.segmentation import extract_ground

        click.echo(f"Loading {input_file}...")
        pcd, header, vlrs, wkt = read(input_file)

        click.echo("Extracting ground...")
        ground_pcd, non_ground_pcd, plane = extract_ground(pcd, distance_threshold=distance_threshold)

        if output is None:
            output = input_file.replace(".", "_ground.")

        save(ground_pcd, output, header=header, vlrs=vlrs, wkt=wkt)
        click.echo(f"Ground: {len(ground_pcd.points):,} points")
        click.echo(f"Non-ground: {len(non_ground_pcd.points):,} points")
        click.echo(f"Plane model: {[f'{c:.4f}' for c in plane]}")
        click.echo(f"Saved to: {output}")

    @main.command()
    @click.argument("input_file")
    @click.option("-r", "--resolution", default=1.0, help="Grid resolution.")
    @click.option("-t", "--dem-type", default="dtm", type=click.Choice(["dtm", "dsm"]))
    @click.option("-o", "--output", default=None, help="Output PNG path.")
    def dtm(input_file: str, resolution: float, dem_type: str, output: str | None):
        """Generate a DTM or DSM and save as PNG."""
        from lidarlens.io import read
        from lidarlens.terrain import generate_dtm, generate_dsm
        from lidarlens.export import dem_to_png

        click.echo(f"Loading {input_file}...")
        pcd, _, _, wkt = read(input_file)

        gen_fn = generate_dtm if dem_type == "dtm" else generate_dsm
        click.echo(f"Generating {dem_type.upper()} (resolution={resolution})...")
        dem_array, meta = gen_fn(pcd, resolution=resolution, wkt=wkt)

        if output is None:
            output = input_file.replace(".", f"_{dem_type}.")

        actual = dem_to_png(dem_array, meta, output)
        click.echo(f"Grid size: {meta['width']} × {meta['height']}")
        click.echo(f"Z range: [{meta['z_min']:.2f}, {meta['z_max']:.2f}]")
        click.echo(f"Saved to: {actual}")

    @main.command()
    @click.argument("input_file")
    @click.option("-i", "--interval", default=5.0, help="Contour interval.")
    @click.option("-o", "--output", default=None, help="Output GeoJSON path.")
    def contours(input_file: str, interval: float, output: str | None):
        """Generate contour lines as GeoJSON."""
        from lidarlens.io import read
        from lidarlens.terrain import generate_dtm, generate_contours

        click.echo(f"Loading {input_file}...")
        pcd, _, _, wkt = read(input_file)

        click.echo("Generating DTM...")
        dem, meta = generate_dtm(pcd, wkt=wkt)

        click.echo(f"Generating contours (interval={interval})...")
        geojson = generate_contours(dem, meta, interval=interval, wkt=wkt)

        if output is None:
            output = input_file.rsplit(".", 1)[0] + "_contours.geojson"

        with open(output, "w") as f:
            json.dump(geojson, f)

        n_features = len(geojson.get("features", []))
        click.echo(f"Generated {n_features} contour lines")
        click.echo(f"Saved to: {output}")

    @main.command("export")
    @click.argument("input_file")
    @click.option("-f", "--format", "fmt", default="ply", type=click.Choice(["csv", "ply", "pcd"]))
    @click.option("-o", "--output", default=None, help="Output file path.")
    def export_cmd(input_file: str, fmt: str, output: str | None):
        """Convert a point cloud to another format."""
        from lidarlens.io import read
        from lidarlens import export

        click.echo(f"Loading {input_file}...")
        pcd, _, _, _ = read(input_file)

        if output is None:
            output = input_file.rsplit(".", 1)[0] + f".{fmt}"

        fn = {"csv": export.to_csv, "ply": export.to_ply, "pcd": export.to_pcd}[fmt]
        actual = fn(pcd, output)
        click.echo(f"Exported to: {actual} ({len(pcd.points):,} points)")

    @main.command()
    @click.argument("source")
    @click.argument("target")
    def compare(source: str, target: str):
        """Compare two point clouds (cloud-to-cloud distance)."""
        from lidarlens.io import read
        from lidarlens.analysis import compare_clouds

        click.echo(f"Loading source: {source}")
        pcd_a, _, _, _ = read(source)
        click.echo(f"Loading target: {target}")
        pcd_b, _, _, _ = read(target)

        click.echo("Computing distances...")
        _, _, stats = compare_clouds(pcd_a, pcd_b)

        click.echo(f"\n{'─' * 40}")
        click.echo(f"  Source points: {stats['source_points']:,}")
        click.echo(f"  Target points: {stats['target_points']:,}")
        click.echo(f"  Min distance:  {stats['min_distance']:.4f}")
        click.echo(f"  Max distance:  {stats['max_distance']:.4f}")
        click.echo(f"  Mean distance: {stats['mean_distance']:.4f}")
        click.echo(f"  Std distance:  {stats['std_distance']:.4f}")
        click.echo(f"  RMSE:          {stats['rmse']:.4f}")
        click.echo(f"{'─' * 40}")
