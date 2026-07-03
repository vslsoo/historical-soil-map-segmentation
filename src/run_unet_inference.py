"""Run the trained U-Net over the whole map (or a test --bbox) and write a
classified GeoTIFF.

Usage:
    python src/run_unet_inference.py [path/to/map.tif] \\
        [--checkpoint output/unet.pt] [--out output/unet_classified.tif] \\
        [--patch-size 128] [--downsample 4] [--bbox ...] [--overlap 0.5] \\
        [--fg-bias 0] [--close-radius 0] [--open-radius 0] \\
        [--min-object-size 0] [--min-hole-size 0] [--vectorize]

--downsample and --patch-size must match what prepare_training_data.py used,
since the model was trained on patches at that resolution.

Tiles are read with --overlap fractional overlap (0.5 = 50%, matching a
tile_size/stride=256/128 setup) and their softmax probabilities are averaged
in the overlapping regions before the final argmax. Non-overlapping tiles
decide independently and produce blocky, jagged seams at tile edges;
overlap-and-average smooths those out, the same way a sliding-window
inference pass with probability averaging is done in most tiled segmentation
pipelines.

--fg-bias shifts the recall/precision balance without retraining (higher ->
catches more area but bleeds past real edges). Pass one number to apply it
to every target class equally, or one number per class (in CATEGORIES
order, e.g. class_10 then class_12) to tune them independently -- e.g.
`--fg-bias 0.2 1.0` barely touches class_10 but pushes class_12 hard.

Cleanup passes run in this order, each optional, all using round (disk)
structuring elements for smoother boundaries than a square/cross element:
  --close-radius fills small holes/gaps that split one real zone into many
    tiny polygons (disk dilate then erode, only ever grows into background).
  --open-radius trims thin ragged spillover at boundaries without shrinking
    genuine blobs (disk erode then dilate, only ever shrinks a mask).
  --min-object-size drops connected blobs smaller than this many pixels.
  --min-hole-size fills holes smaller than this many pixels inside a blob.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio import Affine
from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, disk

from unet_model import build_model

CATEGORIES = ["class_10", "class_12"]  # index 0 in model output is background


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


def tile_positions(total: int, block_size: int, stride: int):
    positions = list(range(0, max(total - block_size, 0) + 1, stride))
    if not positions:
        return [0]
    if positions[-1] != total - block_size:
        positions.append(max(total - block_size, 0))
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiff", nargs="?", help="Path to georeferenced TIFF (defaults to first file in data/raw)")
    parser.add_argument("--checkpoint", default="output/unet.pt")
    parser.add_argument("--out", default="output/unet_classified.tif")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.5, help="Fractional tile overlap for smoother, averaged predictions (0 disables overlap)")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("COL_MIN", "ROW_MIN", "COL_MAX", "ROW_MAX"),
                         help="Only process this pixel region, in source resolution (for quick testing)")
    parser.add_argument("--fg-bias", type=float, nargs="+", default=[0.0],
                         help="Added to target-class logits before softmax; >0 makes the model call more pixels as that class "
                              "(higher recall, lower precision). Pass one value to apply it to every target class, or one "
                              f"value per class in CATEGORIES order ({', '.join(CATEGORIES)}) to tune them independently, "
                              "e.g. --fg-bias 0.2 1.0")
    parser.add_argument("--close-radius", type=int, default=0,
                         help="Disk-shaped morphological closing per class to fill small holes/gaps that split one real zone into many tiny polygons. 0 disables it.")
    parser.add_argument("--open-radius", type=int, default=0,
                         help="Disk-shaped morphological opening per class to trim ragged spillover at boundaries. 0 disables it.")
    parser.add_argument("--min-object-size", type=int, default=0, help="Drop connected blobs smaller than this many px. 0 disables it.")
    parser.add_argument("--min-hole-size", type=int, default=0, help="Fill holes smaller than this many px inside a blob. 0 disables it.")
    parser.add_argument("--vectorize", action="store_true")
    parser.add_argument("--smooth-tolerance", type=float, default=0.0,
                         help="Rounds off the pixel-grid 'staircase' edges left by vectorizing a raster, in map units (meters here). "
                              "Applied as a buffer(+tol).buffer(-tol) pass (rounds convex corners) followed by simplify(tol). "
                              "Try something around 1-2x the output pixel size (downsample * source pixel size). 0 disables it.")
    args = parser.parse_args()

    if len(args.fg_bias) == 1:
        fg_bias = args.fg_bias * len(CATEGORIES)
    elif len(args.fg_bias) == len(CATEGORIES):
        fg_bias = args.fg_bias
    else:
        raise ValueError(f"--fg-bias needs 1 or {len(CATEGORIES)} values (one per class in {CATEGORIES}), got {len(args.fg_bias)}")
    print(f"fg_bias per class: {dict(zip(CATEGORIES, fg_bias))}")

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
    num_classes = checkpoint["num_classes"]
    model = build_model(
        num_classes,
        architecture=checkpoint.get("architecture", "custom"),
        encoder_name=checkpoint.get("encoder_name", "resnet34"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}, using device {device}")

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    block_size = args.patch_size * args.downsample
    stride = max(int(block_size * (1 - args.overlap)), 1)
    fg_bias_tensor = torch.tensor(fg_bias, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    with rasterio.open(tiff_path) as src:
        if args.bbox:
            col_min, row_min, col_max, row_max = args.bbox
            width, height = col_max - col_min, row_max - row_min
            col_offset, row_offset = col_min, row_min
        else:
            width, height = src.width, src.height
            col_offset, row_offset = 0, 0

        base_transform = src.window_transform(Window(col_offset, row_offset, width, height))
        transform = base_transform * Affine.scale(args.downsample)
        out_width = -(-width // args.downsample)
        out_height = -(-height // args.downsample)

        prob_sum = np.zeros((num_classes, out_height, out_width), dtype=np.float32)
        count_sum = np.zeros((out_height, out_width), dtype=np.float32)

        row_positions = tile_positions(height, block_size, stride)
        col_positions = tile_positions(width, block_size, stride)
        total = len(row_positions) * len(col_positions)
        done = 0

        for row_off in row_positions:
            h = min(block_size, height - row_off)
            for col_off in col_positions:
                w = min(block_size, width - col_off)
                read_window = Window(col_offset + col_off, row_offset + row_off, w, h)
                rgb = read_downsampled(src, read_window, args.downsample)
                padded, (orig_h, orig_w) = pad_to_multiple(rgb)

                tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    logits[:, 1:] += fg_bias_tensor
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                probs = probs[:, :orig_h, :orig_w]

                oh, ow = row_off // args.downsample, col_off // args.downsample
                ph, pw = probs.shape[1], probs.shape[2]
                prob_sum[:, oh:oh + ph, ow:ow + pw] += probs
                count_sum[oh:oh + ph, ow:ow + pw] += 1

                done += 1
                print(f"\rTile {done}/{total}", end="", flush=True)
        print()

        prob_map = prob_sum / np.maximum(count_sum, 1)
        working = prob_map.argmax(axis=0).astype(np.uint8)

        for class_id in range(1, num_classes):
            original_mask = working == class_id
            mask = original_mask
            if args.min_object_size > 0:
                mask = remove_small_objects(mask, max_size=args.min_object_size)
            if args.min_hole_size > 0:
                mask = remove_small_holes(mask, max_size=args.min_hole_size)
            if args.close_radius > 0:
                closed = closing(mask, footprint=disk(args.close_radius))
                # closing may dilate into neighboring pixels -- only let it claim background,
                # never steal pixels another class already owns
                mask = closed & ((working == 0) | original_mask)
            if args.open_radius > 0:
                # opening only ever shrinks a mask towards its interior, so it can't
                # newly overlap another class -- safe without the same guard
                mask = opening(mask, footprint=disk(args.open_radius))

            working[original_mask & ~mask] = 0
            working[mask] = class_id

        profile = src.profile.copy()
        profile.update(
            count=1, dtype="uint8", nodata=0,
            width=out_width, height=out_height, transform=transform,
            compress="lzw", tiled=True, blockxsize=256, blockysize=256,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(working[None, :, :])

    print(f"Wrote classified raster to {out_path}")

    if args.vectorize:
        import geopandas as gpd
        from shapely.geometry import shape as shapely_shape

        with rasterio.open(out_path) as src:
            band = src.read(1)
            mask = band != 0
            records = [
                {"class_id": int(value), "category": CATEGORIES[int(value) - 1], "geometry": shapely_shape(geom)}
                for geom, value in shapes(band, mask=mask, transform=src.transform)
            ]
            gdf = gpd.GeoDataFrame.from_records(records)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=src.crs)

        if args.smooth_tolerance > 0:
            tol = args.smooth_tolerance
            gdf["geometry"] = gdf.geometry.buffer(tol).buffer(-tol).simplify(tol, preserve_topology=True)
            gdf = gdf[~gdf.geometry.is_empty]
            print(f"Smoothed polygon edges (tolerance={tol} map units)")

        gpkg_path = out_path.with_suffix(".gpkg")
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"Wrote {len(gdf)} polygons to {gpkg_path}")


if __name__ == "__main__":
    main()
