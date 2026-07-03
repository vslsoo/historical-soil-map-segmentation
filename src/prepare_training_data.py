"""Turn hand-drawn training polygons (from QGIS / Geo-SAM) into (image, label)
patch pairs for U-Net training.

Usage:
    python src/prepare_training_data.py [path/to/map.tif] \\
        [--labels data/labels/training_masks.gpkg] [--out data/labels/patches.npz] \\
        [--patch-size 128] [--downsample 4] [--random-background 20]

Two GeoPackage layouts are supported:
  - Multi-layer (what Geo-SAM produces): one layer per class, named
    "class_10", "class_12", "class_13", or anything else (e.g. "class_6",
    "class_14", "other") for confusable look-alike areas you want the model
    to see as explicit hard negatives. The layer name alone determines the
    class; per-feature attributes are ignored.
  - Single layer with a text attribute (--class-field, default "class")
    holding the same values, if you drew everything into one layer.

Labeled polygons do NOT need to be clustered or cover contiguous ground —
patches are extracted in a --patch-size tile grid, and only tiles that
actually touch a labeled polygon's bounding box are used (plus optional
random background tiles). Pixels inside a used tile that aren't covered by
any polygon are treated as background, so keep polygons reasonably tight to
avoid mislabeling nearby unlabeled ground as background.

Everything is read at --downsample resolution (averaged, same as
classify_map.py) so the labels and the halftone-noisy print pattern line up
at the same smoothing level the model will see at inference time.

Saved as a single patches.npz with `images` (N,H,W,3 uint8) and `labels`
(N,H,W uint8, 0=background/other, 1=class_10, 2=class_12).
"""
import argparse
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio import Affine

TARGET_CLASSES = {"10": 1, "12": 2}


def read_downsampled(dataset, window: Window, downsample: int) -> np.ndarray:
    out_h = max(-(-window.height // downsample), 1)
    out_w = max(-(-window.width // downsample), 1)
    arr = dataset.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w), resampling=Resampling.average)
    return np.moveaxis(arr, 0, -1)


def class_id_from_name(name: str) -> int:
    suffix = name[len("class_"):] if name.startswith("class_") else name
    return TARGET_CLASSES.get(suffix, 0)


def load_labels(path: Path, class_field: str) -> gpd.GeoDataFrame:
    layer_info = pyogrio.list_layers(path)
    layer_names = list(layer_info[:, 0])

    if len(layer_names) > 1:
        frames = []
        for layer_name in layer_names:
            gdf = gpd.read_file(path, layer=layer_name)
            gdf = gdf[["geometry"]].copy()
            gdf["class_id"] = class_id_from_name(layer_name)
            frames.append(gdf)
        combined = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)

    gdf = gpd.read_file(path, layer=layer_names[0] if layer_names else None)
    gdf["class_id"] = gdf[class_field].map(class_id_from_name)
    return gdf[["geometry", "class_id"]]


def tiles_touching(gdf: gpd.GeoDataFrame, transform, tile_px: int, width: int, height: int):
    inv = ~transform
    tiles = set()
    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        col_a, row_a = inv * (minx, maxy)
        col_b, row_b = inv * (maxx, miny)
        col_min, col_max = sorted((col_a, col_b))
        row_min, row_max = sorted((row_a, row_b))
        for tr in range(int(row_min) // tile_px, int(row_max) // tile_px + 1):
            for tc in range(int(col_min) // tile_px, int(col_max) // tile_px + 1):
                if 0 <= tr * tile_px < height and 0 <= tc * tile_px < width:
                    tiles.add((tr, tc))
    return tiles


def tile_window(tile_row: int, tile_col: int, tile_px: int, width: int, height: int) -> Window:
    col_off = min(tile_col * tile_px, max(width - tile_px, 0))
    row_off = min(tile_row * tile_px, max(height - tile_px, 0))
    return Window(col_off, row_off, min(tile_px, width), min(tile_px, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiff", nargs="?", help="Path to georeferenced TIFF (defaults to first file in data/raw)")
    parser.add_argument("--labels", default="data/labels/training_masks.gpkg", help="GeoPackage with training polygons")
    parser.add_argument("--out", default="data/labels/patches.npz", help="Output .npz path")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch side length in downsampled pixels")
    parser.add_argument("--downsample", type=int, default=4, help="Average this many source pixels per side (must match classify_map.py / run_unet_inference.py)")
    parser.add_argument("--random-background", type=int, default=20, help="Extra random background-only patches from anywhere on the map")
    parser.add_argument("--class-field", default="class", help="Attribute field holding the class name, only used for single-layer GeoPackages")
    parser.add_argument("--seed", type=int, default=0)
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

    labels_path = repo_root / args.labels
    gdf = load_labels(labels_path, args.class_field)

    tile_px = args.patch_size * args.downsample
    images, labels = [], []

    with rasterio.open(tiff_path) as src:
        if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        used_tiles = tiles_touching(gdf, src.transform, tile_px, src.width, src.height)
        print(f"{len(gdf)} labeled polygons -> {len(used_tiles)} tiles of {args.patch_size}px (after {args.downsample}x downsample)")

        shapes = [(geom, cid) for geom, cid in zip(gdf.geometry, gdf["class_id"]) if cid > 0]

        class_pixel_counts = {v: 0 for v in TARGET_CLASSES.values()}
        for tile_row, tile_col in sorted(used_tiles):
            window = tile_window(tile_row, tile_col, tile_px, src.width, src.height)
            tile_transform = src.window_transform(window) * Affine.scale(args.downsample)

            image_patch = read_downsampled(src, window, args.downsample)
            label_patch = rasterize(
                shapes, out_shape=image_patch.shape[:2], transform=tile_transform, fill=0, dtype="uint8"
            ) if shapes else np.zeros(image_patch.shape[:2], dtype=np.uint8)

            images.append(image_patch)
            labels.append(label_patch)
            for cid in class_pixel_counts:
                class_pixel_counts[cid] += int((label_patch == cid).sum())

        for cls_name, cid in TARGET_CLASSES.items():
            print(f"  class_{cls_name}: {class_pixel_counts[cid]} px across all tiles")
        print(f"Extracted {len(images)} patches from labeled tiles")

        rng = random.Random(args.seed)
        added = 0
        attempts = 0
        max_tile_row = max(src.height - tile_px, 0) // tile_px
        max_tile_col = max(src.width - tile_px, 0) // tile_px
        while added < args.random_background and attempts < args.random_background * 20:
            attempts += 1
            tr = rng.randint(0, max_tile_row)
            tc = rng.randint(0, max_tile_col)
            if (tr, tc) in used_tiles:
                continue
            window = tile_window(tr, tc, tile_px, src.width, src.height)
            image_patch = read_downsampled(src, window, args.downsample)
            if image_patch.shape[:2] != (args.patch_size, args.patch_size):
                continue
            images.append(image_patch)
            labels.append(np.zeros((args.patch_size, args.patch_size), dtype=np.uint8))
            added += 1
        print(f"Added {added} random background patches")

    images = np.stack(images)
    labels = np.stack(labels)

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=images, labels=labels)
    print(f"Saved {len(images)} patches to {out_path}")


if __name__ == "__main__":
    main()
