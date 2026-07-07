"""Segment the map into closed zone polygons using color-gradient watershed,
independent of fill color/hachure texture and the printed class numbers --
this recovers the drawn boundary network even where a line is faint, worn,
or has small gaps.

Usage:
    python src/extract_zone_boundaries.py [path/to/map.tif] \\
        [--out output/zones.gpkg] [--downsample 2] \\
        [--smooth-ksize 21] [--seed-percentile 15] [--min-seed-size 80] \\
        [--min-zone-size 100] [--bbox COL_MIN ROW_MIN COL_MAX ROW_MAX]

Why watershed instead of thresholding the ink/edges into a binary line mask
and taking connected components of the background: a binary "is this pixel
a boundary" mask always has some real gaps (faint print, a line thinner
than the detector's threshold, scan noise breaking a stroke), and a single
gap silently floods two real zones together into one. Watershed instead
grows from confident interior seeds (flat, low local color-gradient
regions) uphill along a color-gradient "elevation" map and always finds
*some* dividing ridge between two neighboring seeds -- even a very faint
one -- rather than requiring a fully closed line before it will separate
them.

Pipeline:
  1. Median-blur each LAB channel heavily (--smooth-ksize) so the hachure
     fill texture and printed numbers disappear into a flat per-zone
     average color, leaving only genuine zone-to-zone color transitions in
     the gradient.
  2. Sobel gradient magnitude across L/a/b, combined into one "elevation"
     map.
  3. Seeds = connected regions of the lowest --seed-percentile of gradient
     (i.e. the flattest, most uniform interiors), with tiny ones
     (< --min-seed-size) dropped as noise rather than kept as their own
     zone.
  4. skimage.segmentation.watershed floods the elevation map from those
     seeds, producing one integer zone id per pixel with no unassigned
     gaps.
  5. Vectorize to polygons (rasterio.features.shapes), one row per zone,
     with each zone's median RGB attached as an attribute (for later
     linking to legend colors / class identification).

Runs at --downsample (default 2) to keep the whole-map arrays (gradient,
seeds, zones) in memory -- this is a global operation (seeds/watershed need
the whole map at once), unlike the tiled per-block passes in the other
scripts here.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio import Affine
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape as shapely_shape
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def compute_elevation(rgb: np.ndarray, smooth_ksize: int) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    grad = np.zeros(lab.shape[:2], dtype=np.float32)
    for c in range(3):
        smooth = cv2.medianBlur(lab[..., c].astype(np.uint8), smooth_ksize).astype(np.float32)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad += gx ** 2 + gy ** 2
    return np.sqrt(grad)


def segment_zones(elevation: np.ndarray, seed_percentile: float, min_seed_size: int) -> np.ndarray:
    thresh = np.percentile(elevation, seed_percentile)
    flat = elevation < thresh
    seeds = label(flat, connectivity=1)
    seeds = remove_small_objects(seeds, min_size=min_seed_size)
    seeds = label(seeds > 0, connectivity=1)
    return watershed(elevation, markers=seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiff", nargs="?", help="Path to georeferenced TIFF (defaults to first file in data/raw)")
    parser.add_argument("--out", default="output/zones.gpkg")
    parser.add_argument("--downsample", type=int, default=2, help="Read the source at 1/N resolution before segmenting")
    parser.add_argument("--smooth-ksize", type=int, default=21, help="Median blur kernel (odd) used to erase hachure texture before computing the gradient")
    parser.add_argument("--seed-percentile", type=float, default=15.0, help="Percentile of the gradient magnitude below which a pixel counts as a flat 'zone interior' seed")
    parser.add_argument("--min-seed-size", type=int, default=80, help="Drop candidate seed regions smaller than this many px (post-downsample)")
    parser.add_argument("--min-zone-size", type=int, default=100, help="Drop final zones smaller than this many px (post-downsample)")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("COL_MIN", "ROW_MIN", "COL_MAX", "ROW_MAX"),
                         help="Only process this pixel region, in source resolution (for quick testing)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    if args.tiff:
        tiff_path = Path(args.tiff)
    else:
        candidates = sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.tiff"))
        if not candidates:
            raise FileNotFoundError(f"No .tif/.tiff files found in {raw_dir}")
        tiff_path = candidates[0]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tiff_path) as src:
        if args.bbox:
            col_min, row_min, col_max, row_max = args.bbox
            width, height = col_max - col_min, row_max - row_min
            col_off, row_off = col_min, row_min
        else:
            width, height = src.width, src.height
            col_off, row_off = 0, 0
        window = Window(col_off, row_off, width, height)

        out_h = max(-(-height // args.downsample), 1)
        out_w = max(-(-width // args.downsample), 1)
        print(f"Reading {width}x{height} at 1/{args.downsample} -> {out_w}x{out_h}...")
        rgb = np.moveaxis(
            src.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w), resampling=Resampling.average),
            0, -1,
        )
        transform = src.window_transform(window) * Affine.scale(args.downsample)
        crs = src.crs

    print("Computing color-gradient elevation map...")
    elevation = compute_elevation(rgb, args.smooth_ksize)

    print("Segmenting zones via watershed...")
    zones = segment_zones(elevation, args.seed_percentile, args.min_seed_size)
    zones = remove_small_objects(zones, min_size=args.min_zone_size).astype(np.int32)
    print(f"Found {zones.max()} zones")

    print("Computing per-zone median color...")
    zone_ids = np.arange(1, zones.max() + 1)
    median_rgb = np.stack([
        ndimage.median(rgb[..., c], labels=zones, index=zone_ids) for c in range(3)
    ], axis=-1)

    print("Vectorizing...")
    records = []
    for geom, value in shapes(zones, mask=zones != 0, transform=transform):
        zone_id = int(value)
        r, g, b = median_rgb[zone_id - 1]
        records.append({
            "zone_id": zone_id,
            "median_r": round(float(r)), "median_g": round(float(g)), "median_b": round(float(b)),
            "geometry": shapely_shape(geom),
        })
    gdf = gpd.GeoDataFrame.from_records(records)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)
    gdf.to_file(out_path, driver="GPKG")
    print(f"Wrote {len(gdf)} zone polygons to {out_path}")


if __name__ == "__main__":
    main()
