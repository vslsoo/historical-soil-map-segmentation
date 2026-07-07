"""Denoise, contrast-enhance, and color-balance a scanned map raster before
running the rest of the pipeline on it.

Usage:
    python src/preprocess_raster.py [path/to/map.tif] \\
        [--out data/raw/raster_enhanced.tif] [--median-ksize 3] \\
        [--clip-percent 1.0] [--clahe-clip 2.0] [--clahe-grid 8] \\
        [--block-size 2048] [--bbox COL_MIN ROW_MIN COL_MAX ROW_MAX]

Three passes, in order, per tile:
  1. Median blur -- knocks out the salt-and-pepper speckle/spots that come
     from scanning old printed paper, without blurring real edges much (unlike
     a Gaussian blur). Keep --median-ksize small (3 or 5): the map's fill
     patterns are thin hachures the rest of the pipeline depends on, and a
     bigger kernel erases them along with the noise.
  2. Per-channel percentile stretch -- clips the darkest/lightest
     --clip-percent of pixels (computed once, globally, from a fast decimated
     pass over the whole raster) and rescales each channel independently to
     the full 0-255 range. Stretching channels independently is what fixes
     color balance: a uniform yellow/sepia cast means one channel (usually
     blue) never reaches full range, so stretching it out separately
     re-centers the whites/blacks per channel. Because the stats are global,
     there's no brightness drift between tiles.
  3. CLAHE on the L channel only (LAB colorspace) -- adds local contrast
     without reintroducing the color cast that per-channel CLAHE would.

Any single pass can be disabled by zeroing its parameter
(--median-ksize 0 / --clip-percent 0 / --clahe-clip 0).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling


def compute_percentiles(src, window: Window, clip_percent: float, sample_downsample: int = 16):
    out_h = max(int(window.height) // sample_downsample, 1)
    out_w = max(int(window.width) // sample_downsample, 1)
    sample = src.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w), resampling=Resampling.average)
    sample = np.moveaxis(sample, 0, -1)
    low = np.percentile(sample, clip_percent, axis=(0, 1))
    high = np.percentile(sample, 100 - clip_percent, axis=(0, 1))
    return low, high


def stretch(tile: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    out = np.empty_like(tile, dtype=np.float32)
    for c in range(3):
        span = max(high[c] - low[c], 1.0)
        out[..., c] = (tile[..., c].astype(np.float32) - low[c]) * (255.0 / span)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe_l(tile: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def tile_positions(total: int, block_size: int):
    return list(range(0, total, block_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiff", nargs="?", help="Path to georeferenced TIFF (defaults to first file in data/raw)")
    parser.add_argument("--out", default="data/raw/raster_enhanced.tif")
    parser.add_argument("--median-ksize", type=int, default=3, help="Median blur kernel size (odd, 0 disables denoising)")
    parser.add_argument("--clip-percent", type=float, default=1.0,
                         help="Percent of darkest/lightest pixels per channel clipped before rescaling to 0-255 (0 disables the stretch)")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit on the L channel (0 disables CLAHE)")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size")
    parser.add_argument("--block-size", type=int, default=2048, help="Processing tile size in source pixels")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("COL_MIN", "ROW_MIN", "COL_MAX", "ROW_MAX"),
                         help="Only process this pixel region (for a quick before/after preview)")
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
            col_off0, row_off0 = col_min, row_min
        else:
            width, height = src.width, src.height
            col_off0, row_off0 = 0, 0
        base_window = Window(col_off0, row_off0, width, height)

        if args.clip_percent > 0:
            print("Computing global per-channel percentiles...")
            low, high = compute_percentiles(src, base_window, args.clip_percent)
            print(f"  low={low.round(1).tolist()} high={high.round(1).tolist()}")
        else:
            low = high = None

        transform = src.window_transform(base_window)
        profile = src.profile.copy()
        profile.update(
            width=width, height=height, transform=transform,
            compress="lzw", tiled=True, blockxsize=256, blockysize=256,
        )

        row_positions = tile_positions(height, args.block_size)
        col_positions = tile_positions(width, args.block_size)
        total = len(row_positions) * len(col_positions)
        done = 0

        with rasterio.open(out_path, "w", **profile) as dst:
            for row_off in row_positions:
                h = min(args.block_size, height - row_off)
                for col_off in col_positions:
                    w = min(args.block_size, width - col_off)
                    window = Window(col_off0 + col_off, row_off0 + row_off, w, h)
                    tile = np.ascontiguousarray(np.moveaxis(src.read([1, 2, 3], window=window), 0, -1))

                    if args.median_ksize > 1:
                        tile = cv2.medianBlur(tile, args.median_ksize)
                    if low is not None:
                        tile = stretch(tile, low, high)
                    if args.clahe_clip > 0:
                        tile = apply_clahe_l(tile, args.clahe_clip, args.clahe_grid)

                    dst.write(np.moveaxis(tile, -1, 0), window=Window(col_off, row_off, w, h))

                    done += 1
                    print(f"\rTile {done}/{total}", end="", flush=True)
        print()

    print(f"Wrote enhanced raster to {out_path}")


if __name__ == "__main__":
    main()
