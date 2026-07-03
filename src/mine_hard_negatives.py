"""Automatically mine hard-negative training polygons from the model's own
predictions, instead of manually tracing thin features like rivers,
boundaries, or railways.

The idea: real soil-zone polygons are blobby (many pixels wide), while false
positives on rivers/boundaries/grid lines/railways are geometrically thin
(a few pixels wide). Morphological opening (erode then dilate) makes thin
structures vanish while leaving blobby ones intact -- so whatever the model
predicted that does NOT survive opening is, geometrically, a thin
line-like artifact. Those pixels are vectorized straight into a new
GeoPackage layer that prepare_training_data.py already treats as an
explicit "other" hard negative (any layer name other than class_10/class_12
counts as background/other).

Usage:
    python src/mine_hard_negatives.py [path/to/map.tif] \\
        [--checkpoint output/unet.pt] [--bbox COL_MIN ROW_MIN COL_MAX ROW_MAX] \\
        [--thinness-radius 3] [--min-size 4] \\
        [--labels data/labels/training_masks.gpkg] [--layer class_auto_line]

Caveat: this flags predictions by shape alone, so a genuinely thin real
zone (a narrow valley strip, say) could get miscaptured as a negative too --
skim the output layer in QGIS before trusting it blindly, especially the
first few times.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape
from scipy.ndimage import binary_opening

from unet_model import build_model

CATEGORIES = ["class_10", "class_12"]


def read_downsampled(dataset, window: Window, downsample: int) -> np.ndarray:
    out_h = max(-(-window.height // downsample), 1)
    out_w = max(-(-window.width // downsample), 1)
    arr = dataset.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w), resampling=Resampling.average)
    return np.moveaxis(arr, 0, -1)


def pad_to_multiple(arr: np.ndarray, multiple: int = 8):
    h, w = arr.shape[:2]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return arr, (h, w)
    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, (h, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiff", nargs="?", help="Path to georeferenced TIFF (defaults to first file in data/raw)")
    parser.add_argument("--checkpoint", default="output/unet.pt")
    parser.add_argument("--bbox", type=int, nargs=4, required=True, metavar=("COL_MIN", "ROW_MIN", "COL_MAX", "ROW_MAX"),
                         help="Region to mine, in source pixel resolution")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--thinness-radius", type=int, default=3,
                         help="Opening radius (downsampled px) used to decide 'thin' -- bigger = stricter, flags more as negatives")
    parser.add_argument("--min-size", type=int, default=4,
                         help="Ignore fragments smaller than this many pixels (pure noise, not worth a training polygon)")
    parser.add_argument("--labels", default="data/labels/training_masks.gpkg")
    parser.add_argument("--layer", default="class_auto_line", help="Layer name to write/append to (must not be class_10/class_12)")
    parser.add_argument("--all-predictions-negative", action="store_true",
                         help="Treat EVERY predicted target-class pixel in --bbox as a hard negative, not just the thin/line-like "
                              "fraction. Only use this over a region you've manually verified has no real class_10/12 at all -- "
                              "e.g. a totally different climate zone the training data never covered.")
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

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    checkpoint = torch.load(repo_root / args.checkpoint, map_location=device)
    model = build_model(
        checkpoint["num_classes"],
        architecture=checkpoint.get("architecture", "custom"),
        encoder_name=checkpoint.get("encoder_name", "resnet34"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    col_min, row_min, col_max, row_max = args.bbox
    width, height = col_max - col_min, row_max - row_min
    block_size = args.patch_size * args.downsample

    with rasterio.open(tiff_path) as src:
        window = Window(col_min, row_min, width, height)
        transform = src.window_transform(window) * rasterio.Affine.scale(args.downsample)
        out_w, out_h = -(-width // args.downsample), -(-height // args.downsample)
        band = np.zeros((out_h, out_w), dtype=np.uint8)

        for row_off in range(0, height, block_size):
            h = min(block_size, height - row_off)
            for col_off in range(0, width, block_size):
                w = min(block_size, width - col_off)
                rgb = read_downsampled(src, Window(col_min + col_off, row_min + row_off, w, h), args.downsample)
                padded, (orig_h, orig_w) = pad_to_multiple(rgb)
                tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred = pred[:orig_h, :orig_w]
                oh, ow = row_off // args.downsample, col_off // args.downsample
                band[oh:oh + pred.shape[0], ow:ow + pred.shape[1]] = pred[:band.shape[0] - oh, :band.shape[1] - ow]

    print(f"Predicted {sum((band == cid).sum() for cid in range(1, len(CATEGORIES) + 1))} target-class px over the region")

    thin_mask = np.zeros_like(band, dtype=bool)
    for class_id in range(1, len(CATEGORIES) + 1):
        mask = band == class_id
        if args.all_predictions_negative:
            thin_mask |= mask
            print(f"{CATEGORIES[class_id - 1]}: {mask.sum()} predicted px, all flagged as negative (--all-predictions-negative)")
        else:
            opened = binary_opening(mask, iterations=args.thinness_radius)
            thin_mask |= mask & ~opened
            print(f"{CATEGORIES[class_id - 1]}: {mask.sum()} predicted px, {(mask & ~opened).sum()} flagged as thin/line-like")

    records = [
        {"geometry": shapely_shape(geom)}
        for geom, value in shapes(thin_mask.astype(np.uint8), mask=thin_mask, transform=transform)
        if value == 1
    ]
    # drop tiny slivers below --min-size (in downsampled px^2)
    pixel_area = abs(transform.a * transform.e)
    records = [r for r in records if r["geometry"].area >= args.min_size * pixel_area]

    print(f"Mined {len(records)} thin/line-like polygons (min_size={args.min_size}px)")
    if not records:
        print("Nothing to write.")
        return

    with rasterio.open(tiff_path) as src:
        crs = src.crs
    gdf = gpd.GeoDataFrame.from_records(records)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)

    labels_path = repo_root / args.labels
    if labels_path.exists():
        existing_layers = list(pyogrio.list_layers(labels_path)[:, 0])
        if args.layer in existing_layers:
            existing = gpd.read_file(labels_path, layer=args.layer)
            gdf = gpd.GeoDataFrame(
                pd.concat([existing[["geometry"]], gdf[["geometry"]]], ignore_index=True),
                geometry="geometry", crs=gdf.crs,
            )

    gdf.to_file(labels_path, layer=args.layer, driver="GPKG")
    print(f"Wrote {len(gdf)} total polygons to layer '{args.layer}' in {labels_path}")


if __name__ == "__main__":
    main()
