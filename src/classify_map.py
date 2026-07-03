"""Classify a huge georeferenced map raster into a handful of legend classes
by color distance, processed in tiles so the whole file never has to fit in
memory.

Usage:
    python src/classify_map.py [path/to/map.tif] \\
        [--legend data/legend_samples] [--out output/classified_map.tif] \\
        [--threshold 12] [--downsample 4] [--block-size 2048] [--vectorize]

Old lithographed maps render "solid" legend colors as a fine halftone/hachure
print pattern, so at full pixel resolution a single pixel can be much lighter
or darker than the color it represents (even plain paper can look close to a
light legend color). --downsample averages each NxN block of source pixels
(via GDAL's average resampling) before classification, which smooths the
print pattern out to its true tone. Each full-res block is read and
downsampled independently, so memory use stays bounded regardless of image
size.

Legend reference colors are the median RGB of each patch in
data/legend_samples/manifest.json (median is robust to the small dark label
digit printed inside each swatch). For every (downsampled) pixel, the nearest
legend color in Lab space is found; if it's farther than --threshold (roughly
a perceptual color-difference unit) the pixel is left unclassified (0).

Output is a single-band GeoTIFF, at the downsampled resolution, with the same
CRS as the input: 0 = unclassified, 1..N = legend classes in the order they
appear in the manifest. Pass --vectorize to additionally write polygons to a
GeoPackage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import shapes
from skimage.color import rgb2lab
from PIL import Image


def load_reference_colors(legend_dir: Path):
    manifest = json.loads((legend_dir / "manifest.json").read_text())
    categories = []
    ref_rgb = []
    for entry in manifest:
        patch = np.array(Image.open(legend_dir / entry["file"]).convert("RGB"))
        median_rgb = np.median(patch.reshape(-1, 3), axis=0)
        categories.append(entry["category"])
        ref_rgb.append(median_rgb)
    ref_rgb = np.array(ref_rgb, dtype=np.float64)
    ref_lab = rgb2lab((ref_rgb / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    return categories, ref_lab


def iter_blocks(width, height, block_size):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def read_downsampled(dataset, window: Window, downsample: int) -> np.ndarray:
    out_h = -(-window.height // downsample)
    out_w = -(-window.width // downsample)
    arr = dataset.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w), resampling=Resampling.average)
    return np.moveaxis(arr, 0, -1)


def classify_block(rgb_block: np.ndarray, ref_lab: np.ndarray, threshold: float) -> np.ndarray:
    lab_block = rgb2lab(rgb_block.astype(np.float32) / 255.0).astype(np.float32)
    h, w, _ = lab_block.shape
    flat = lab_block.reshape(-1, 3)

    dists = np.linalg.norm(flat[:, None, :] - ref_lab[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(flat)), nearest]

    labels = np.where(nearest_dist <= threshold, nearest + 1, 0).astype(np.uint8)
    return labels.reshape(h, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiff", nargs="?", help="Path to georeferenced TIFF (defaults to first file in data/raw)")
    parser.add_argument("--legend", default="data/legend_samples", help="Directory with legend patches + manifest.json")
    parser.add_argument("--out", default="output/classified_map.tif", help="Output GeoTIFF path")
    parser.add_argument("--threshold", type=float, default=12.0, help="Max Lab color distance to count as a match")
    parser.add_argument("--downsample", type=int, default=4,
                         help="Average this many source pixels per side before classifying (removes halftone print noise)")
    parser.add_argument("--block-size", type=int, default=2048,
                         help="Source-resolution tile size in pixels for streaming processing (must be a multiple of --downsample)")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("COL_MIN", "ROW_MIN", "COL_MAX", "ROW_MAX"),
                         help="Only process this pixel region, in source resolution (for quick testing before a full run)")
    parser.add_argument("--vectorize", action="store_true", help="Also write classified polygons to a GeoPackage")
    args = parser.parse_args()

    if args.block_size % args.downsample != 0:
        raise ValueError("--block-size must be a multiple of --downsample")

    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    if args.tiff:
        tiff_path = Path(args.tiff)
    else:
        candidates = sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.tiff"))
        if not candidates:
            raise FileNotFoundError(f"No .tif/.tiff files found in {raw_dir}")
        tiff_path = candidates[0]

    legend_dir = repo_root / args.legend
    categories, ref_lab = load_reference_colors(legend_dir)
    print(f"Legend classes: {categories}")

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tiff_path) as src:
        if args.bbox:
            col_min, row_min, col_max, row_max = args.bbox
            width, height = col_max - col_min, row_max - row_min
            col_offset, row_offset = col_min, row_min
        else:
            width, height = src.width, src.height
            col_offset, row_offset = 0, 0

        base_transform = src.window_transform(Window(col_offset, row_offset, width, height))
        transform = base_transform * rasterio.Affine.scale(args.downsample)
        out_width = -(-width // args.downsample)
        out_height = -(-height // args.downsample)

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            width=out_width,
            height=out_height,
            transform=transform,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            total_blocks = ((width + args.block_size - 1) // args.block_size) * \
                           ((height + args.block_size - 1) // args.block_size)
            done = 0
            for src_window in iter_blocks(width, height, args.block_size):
                read_window = Window(
                    col_offset + src_window.col_off,
                    row_offset + src_window.row_off,
                    src_window.width,
                    src_window.height,
                )
                rgb_block = read_downsampled(src, read_window, args.downsample)
                labels = classify_block(rgb_block, ref_lab, args.threshold)

                out_window = Window(
                    src_window.col_off // args.downsample,
                    src_window.row_off // args.downsample,
                    labels.shape[1],
                    labels.shape[0],
                )
                dst.write(labels[None, :, :], window=out_window)
                done += 1
                print(f"\rBlock {done}/{total_blocks}", end="", flush=True)
            print()

    print(f"Wrote classified raster to {out_path}")

    if args.vectorize:
        import geopandas as gpd
        from shapely.geometry import shape as shapely_shape

        with rasterio.open(out_path) as src:
            band = src.read(1)
            mask = band != 0
            records = [
                {"class_id": int(value), "category": categories[int(value) - 1], "geometry": shapely_shape(geom)}
                for geom, value in shapes(band, mask=mask, transform=src.transform)
            ]
            gdf = gpd.GeoDataFrame.from_records(records)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=src.crs)

        gpkg_path = out_path.with_suffix(".gpkg")
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"Wrote {len(gdf)} polygons to {gpkg_path}")


if __name__ == "__main__":
    main()
