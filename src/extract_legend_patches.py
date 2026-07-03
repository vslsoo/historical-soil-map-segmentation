"""Interactively extract legend swatches (color + hatching) from a georeferenced map TIFF.

Usage:
    python src/extract_legend_patches.py [path/to/map.tif] [--out data/legend_samples] [--max-display 1400]

Navigate:
    - Scroll wheel: zoom in/out, centered on the cursor. Each zoom re-reads the
      raster at full resolution for the visible area, so you always see real
      pixels rather than a blurry upsampled thumbnail.
    - Toolbar "pan" (hand icon): drag to move around while zoomed in. Toggle it
      off again afterwards.

Select a legend patch:
    - With the toolbar pan/zoom tool OFF, drag a rectangle over a legend
      swatch with the mouse, release, then type the category name in the
      terminal and press Enter (leave blank to skip a selection).
    - Press 'q' with the plot window focused, or close it, to finish.

Patches are saved as PNGs plus a manifest.json recording the category and the
full-resolution pixel bbox each patch came from.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image


def find_default_tiff(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.tiff"))
    if not candidates:
        raise FileNotFoundError(f"No .tif/.tiff files found in {raw_dir}")
    return candidates[0]


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    max_val = arr.max()
    if max_val <= 0:
        return arr.astype(np.uint8)
    return (arr / max_val * 255).astype(np.uint8)


def read_rgb(dataset: rasterio.io.DatasetReader, window: Window, out_shape=None) -> np.ndarray:
    kwargs = {"window": window}
    if out_shape is not None:
        kwargs["out_shape"] = (3, *out_shape) if dataset.count >= 3 else out_shape

    if dataset.count >= 3:
        arr = dataset.read([1, 2, 3], **kwargs)
        arr = np.moveaxis(arr, 0, -1)
    else:
        band = dataset.read(1, **kwargs)
        colormap = None
        try:
            colormap = dataset.colormap(1)
        except ValueError:
            pass
        if colormap:
            lut = np.zeros((256, 3), dtype=np.uint8)
            for idx, rgba in colormap.items():
                lut[idx] = rgba[:3]
            arr = lut[band]
        else:
            arr = np.stack([band] * 3, axis=-1)

    return to_uint8(arr)


class RasterViewer:
    """Shows the current axes view at native resolution by re-reading the
    raster window on every zoom/pan, instead of stretching a fixed low-res
    thumbnail."""

    def __init__(self, dataset: rasterio.io.DatasetReader, ax, max_display: int):
        self.dataset = dataset
        self.ax = ax
        self.max_display = max_display
        self.width = dataset.width
        self.height = dataset.height
        self._redrawing = False

        arr = self._render(0, 0, self.width, self.height)
        self.im = ax.imshow(arr, extent=(0, self.width, self.height, 0))
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)

        ax.callbacks.connect("xlim_changed", self._on_lims_changed)
        ax.callbacks.connect("ylim_changed", self._on_lims_changed)

    def _render(self, col_min, row_min, col_max, row_max):
        width_px = max(col_max - col_min, 1)
        height_px = max(row_max - row_min, 1)
        scale = max(width_px / self.max_display, height_px / self.max_display, 1.0)
        out_shape = (max(int(height_px / scale), 1), max(int(width_px / scale), 1))
        window = Window(col_min, row_min, width_px, height_px)
        return read_rgb(self.dataset, window, out_shape=out_shape)

    def _on_lims_changed(self, _):
        if self._redrawing:
            return
        self._redrawing = True
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            col_min, col_max = sorted(xlim)
            row_min, row_max = sorted(ylim)
            col_min = int(max(col_min, 0))
            row_min = int(max(row_min, 0))
            col_max = int(min(col_max, self.width))
            row_max = int(min(row_max, self.height))
            if col_max - col_min < 2 or row_max - row_min < 2:
                return
            arr = self._render(col_min, row_min, col_max, row_max)
            self.im.set_data(arr)
            self.im.set_extent((col_min, col_max, row_max, row_min))
            self.ax.figure.canvas.draw_idle()
        finally:
            self._redrawing = False


def make_scroll_zoom_handler(ax, viewer: RasterViewer, zoom_factor: float = 0.8):
    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        factor = zoom_factor if event.button == "up" else 1 / zoom_factor

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x, y = event.xdata, event.ydata

        new_xlim = (x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor)
        new_ylim = (y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor)

        new_xlim = (max(new_xlim[0], 0), min(new_xlim[1], viewer.width))
        new_ylim = (min(new_ylim[0], viewer.height), max(new_ylim[1], 0))

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw_idle()

    return on_scroll


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiff", nargs="?", help="Path to georeferenced TIFF (defaults to first file in data/raw)")
    parser.add_argument("--out", default="data/legend_samples", help="Output directory for patches")
    parser.add_argument("--max-display", type=int, default=1400, help="Max width/height of the rendered view in pixels")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    tiff_path = Path(args.tiff) if args.tiff else find_default_tiff(raw_dir)
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    category_counts = {}
    for entry in manifest:
        category_counts[entry["category"]] = category_counts.get(entry["category"], 0) + 1

    dataset = rasterio.open(tiff_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    viewer = RasterViewer(dataset, ax, args.max_display)
    ax.set_title(
        "Scroll to zoom. Toolbar pan (hand icon) to move, then toggle it off.\n"
        "With pan/zoom off: drag a rectangle over a legend swatch, name it in the terminal.\n"
        "Press 'q' to finish."
    )

    def on_select(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        col_min, col_max = sorted((int(x0), int(x1)))
        row_min, row_max = sorted((int(y0), int(y1)))
        col_min, row_min = max(col_min, 0), max(row_min, 0)
        col_max = min(col_max, dataset.width)
        row_max = min(row_max, dataset.height)
        if col_max - col_min < 2 or row_max - row_min < 2:
            print("Selection too small, skipped.")
            return

        window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
        patch = read_rgb(dataset, window)

        category = input("Category name for this patch (blank to skip): ").strip()
        if not category:
            print("Skipped.")
            return

        category_counts[category] = category_counts.get(category, 0) + 1
        idx = category_counts[category]
        safe_name = category.replace(" ", "_")
        filename = f"{safe_name}_{idx:02d}.png"
        Image.fromarray(patch).save(out_dir / filename)

        manifest.append({
            "file": filename,
            "category": category,
            "bbox_full_res": [col_min, row_min, col_max, row_max],
        })
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"Saved {filename} ({col_max - col_min}x{row_max - row_min}px)")

    selector = RectangleSelector(
        ax, on_select, useblit=True,
        button=[1], minspanx=2, minspany=2,
        spancoords="pixels",
    )

    fig.canvas.mpl_connect("scroll_event", make_scroll_zoom_handler(ax, viewer))

    def on_key(event):
        if event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    dataset.close()
    print(f"\nDone. {len(manifest)} patches saved to {out_dir}")


if __name__ == "__main__":
    main()
