# Historical Soil Map Segmentation

Tools for turning a scanned, georeferenced historical soil map (a huge
GeoTIFF) into vector polygons for one or more soil classes, using a U-Net
trained on hand-labeled examples.

Historical maps like this print each soil zone as a color/hachure fill
shared across many different classes, with the actual class identified by a
small printed number inside each polygon. Because the fill patterns
themselves are often visually similar across classes, the pipeline leans
heavily on **hand-labeled positive and negative (look-alike) examples** and
iterative error correction rather than relying on color alone.

## Pipeline overview

1. **Extract legend samples** — interactively crop reference swatches from
   the map's legend (`src/extract_legend_patches.py`).
2. **Label training polygons in QGIS** — draw polygons for your target
   class(es) and for confusable look-alike classes, using whatever tool you
   like (e.g. the Geo-SAM plugin speeds this up a lot). Save everything into
   one GeoPackage, one layer per class, named `class_<id>`.
3. **Prepare training patches** — rasterizes your labeled polygons against
   the source GeoTIFF and cuts them into fixed-size image/label patches
   (`src/prepare_training_data.py`).
4. **Train** a U-Net (from scratch, or with a pretrained encoder via
   `segmentation_models_pytorch`) on those patches (`src/train_unet.py`).
5. **Run inference** over the whole map (or a test region) with tiled,
   overlap-averaged, optionally test-time-augmented prediction, plus
   morphological cleanup and vectorization to GeoPackage
   (`src/run_unet_inference.py`).
6. **Mine hard negatives/positives** — run the current model over new areas,
   automatically flag likely mistakes (e.g. thin line-like false positives,
   or an entire region known to have no real target class), and feed them
   back into your labeled GeoPackage as extra training polygons
   (`src/mine_hard_negatives.py`). Repeat steps 3-6 as needed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your georeferenced map in `data/raw/` (a single `.tif`).

## Basic usage

```bash
# 1. Crop legend reference patches (interactive)
python3 src/extract_legend_patches.py

# 2. (label polygons in QGIS, save to data/labels/training_masks.gpkg)

# 3. Build training patches from your labeled GeoPackage
python3 src/prepare_training_data.py

# 4. Train
python3 src/train_unet.py --out output/unet.pt

# 5. Classify the whole map and export polygons
python3 src/run_unet_inference.py --checkpoint output/unet.pt --vectorize

# 6. Find likely mistakes in a new region and add them back as training data
python3 src/mine_hard_negatives.py --bbox COL_MIN ROW_MIN COL_MAX ROW_MAX --checkpoint output/unet.pt
```

Each script's `--help` (or its module docstring) documents its own options
in more detail — there are knobs for downsampling factor, tile overlap,
test-time augmentation, recall/precision bias, and morphological cleanup
(hole filling, small-object removal, edge smoothing).

## Repository layout

- `src/` — all pipeline scripts.
- `data/raw/` — your source GeoTIFF (not tracked in git; too large).
- `data/labels/training_masks.gpkg` — hand-labeled training polygons.
- `data/legend_samples/` — cropped legend reference patches.
- `output/` — generated checkpoints, classified rasters, and vector results
  (not tracked in git; regenerate via the scripts above).
