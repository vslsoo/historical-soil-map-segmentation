"""Train the U-Net on patches produced by prepare_training_data.py.

Usage:
    python src/train_unet.py [--patches data/labels/patches.npz] \\
        [--out output/unet.pt] [--epochs 60] [--batch-size 8] [--val-fraction 0.15]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from unet_model import build_model
from losses import CombinedLoss


class PatchDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        lbl = self.labels[idx].astype(np.int64)

        if self.augment:
            if np.random.rand() < 0.5:
                img, lbl = img[:, ::-1, :].copy(), lbl[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img, lbl = img[::-1, :, :].copy(), lbl[::-1, :].copy()
            k = np.random.randint(4)
            img, lbl = np.rot90(img, k).copy(), np.rot90(lbl, k).copy()

        img = torch.from_numpy(img).permute(2, 0, 1)
        lbl = torch.from_numpy(lbl)
        return img, lbl


def stratified_split(labels: np.ndarray, val_fraction: float, num_classes: int, seed: int):
    """Random split, but guarantees every target class that has any patches gets a
    proportional share of them in validation too -- a plain random split can easily
    leave a rare class (e.g. only a handful of patches) entirely out of val, which
    then reports a meaningless 0.0/nan for it every epoch instead of a real score."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    val_mask = np.zeros(n, dtype=bool)

    for c in range(1, num_classes):
        class_patch_idx = np.where((labels == c).any(axis=(1, 2)))[0]
        if len(class_patch_idx) == 0:
            continue
        n_val_c = max(1, int(round(len(class_patch_idx) * val_fraction)))
        chosen = rng.choice(class_patch_idx, size=min(n_val_c, len(class_patch_idx)), replace=False)
        val_mask[chosen] = True

    target_n_val = max(int(round(n * val_fraction)), 1)
    remaining_needed = target_n_val - int(val_mask.sum())
    if remaining_needed > 0:
        remaining_idx = np.where(~val_mask)[0]
        chosen = rng.choice(remaining_idx, size=min(remaining_needed, len(remaining_idx)), replace=False)
        val_mask[chosen] = True

    return np.where(~val_mask)[0], np.where(val_mask)[0]


def compute_class_weights(labels: np.ndarray, num_classes: int, fg_boost: float = 1.0, weight_power: float = 1.0) -> torch.Tensor:
    """weight_power < 1 softens the inverse-frequency ratio (e.g. 0.5 = sqrt), which matters a lot
    here: class_10/12 are so rare relative to background that raw inverse-frequency weighting can
    make "catch every target pixel" outweigh "correctly reject a background/lookalike pixel" by
    100x+, so the model has little incentive to actually learn to reject confusable classes even
    when they're well represented in training tiles."""
    counts = np.bincount(labels.reshape(-1), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = (counts.sum() / (num_classes * counts)) ** weight_power
    weights[1:] *= fg_boost  # extra push towards calling target classes -> favors recall over precision
    return torch.tensor(weights, dtype=torch.float32)


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    ious = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c, target_c = pred == c, target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious[c] = intersection / union if union > 0 else float("nan")
    return ious


def recall_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    recalls = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c, target_c = pred == c, target == c
        tp = (pred_c & target_c).sum().item()
        actual = target_c.sum().item()
        recalls[c] = tp / actual if actual > 0 else float("nan")
    return recalls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", default="data/labels/patches.npz")
    parser.add_argument("--out", default="output/unet.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine",
                         help="'cosine' smoothly decays the learning rate to ~0 over all epochs, which tends to stabilize the "
                              "wild epoch-to-epoch metric swings a small dataset + fixed high LR produces")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--smooth-window", type=int, default=5,
                         help="Select the checkpoint by a moving average of the score over this many epochs, instead of a single epoch's (noisy) value")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--fg-weight-boost", type=float, default=2.0,
                         help="Extra multiplier on target-class loss weights, on top of inverse-frequency weighting -> favors recall over precision")
    parser.add_argument("--weight-power", type=float, default=1.0,
                         help="Softens inverse-frequency class weighting (e.g. 0.5 = sqrt). Rare target classes vs. huge background "
                              "counts can otherwise make missing a target pixel weigh 100x+ more than wrongly including a lookalike "
                              "background pixel, so the model has little incentive to learn to reject confusable classes")
    parser.add_argument("--architecture", choices=["custom", "smp"], default="custom",
                         help="'custom' is the small from-scratch UNet; 'smp' uses an ImageNet-pretrained encoder (segmentation_models_pytorch), which usually generalizes better with little data")
    parser.add_argument("--encoder-name", default="resnet34",
                         help="Encoder backbone when --architecture smp (e.g. resnet18, resnet34, efficientnet-b0)")
    parser.add_argument("--ce-weight", type=float, default=0.5,
                         help="Weight of the (class-weighted) cross-entropy term in the loss")
    parser.add_argument("--tversky-weight", type=float, default=0.5,
                         help="Weight of the Tversky term (per-class overlap on class_10/12 only, ignoring background)")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="Tversky penalty on false positives")
    parser.add_argument("--tversky-beta", type=float, default=0.7,
                         help="Tversky penalty on false negatives; beta > alpha pushes the loss itself towards recall, not just class weights")
    parser.add_argument("--select-by", choices=["iou", "min_recall", "mean_recall"], default="min_recall",
                         help="Checkpoint-selection metric: 'iou' balances precision+recall, 'min_recall' maximizes the worst-performing "
                              "class's recall (matches a 'recall 90+ in every class' goal), 'mean_recall' maximizes average recall")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data = np.load(repo_root / args.patches)
    images, labels = data["images"], data["labels"]
    print(f"Loaded {len(images)} patches, shape {images.shape[1:]}")

    class_names = ["background/other", "class_10", "class_12"]

    torch.manual_seed(args.seed)
    train_idx, val_idx = stratified_split(labels, args.val_fraction, args.num_classes, args.seed)
    n_train, n_val = len(train_idx), len(val_idx)
    train_set = Subset(PatchDataset(images, labels, augment=True), train_idx)
    val_set = Subset(PatchDataset(images, labels, augment=False), val_idx)
    print(f"Train/val split: {n_train} train, {n_val} val patches "
          f"(val contains class_10: {(labels[val_idx] == 1).any()}, class_12: {(labels[val_idx] == 2).any()})")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = build_model(args.num_classes, architecture=args.architecture, encoder_name=args.encoder_name).to(device)
    print(f"Architecture: {args.architecture}" + (f" (encoder={args.encoder_name})" if args.architecture == "smp" else ""))
    class_weights = compute_class_weights(labels, args.num_classes, fg_boost=args.fg_weight_boost, weight_power=args.weight_power).to(device)
    print(f"Class weights (fg_boost={args.fg_weight_boost}): {dict(zip(class_names, class_weights.tolist()))}")
    criterion = CombinedLoss(
        class_weights, args.num_classes,
        ce_weight=args.ce_weight, tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
    )
    print(f"Loss: {args.ce_weight}*CE + {args.tversky_weight}*Tversky(alpha={args.tversky_alpha}, beta={args.tversky_beta})")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.lr_schedule == "cosine" else None

    best_score = -1.0
    score_history = []
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train
        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss = 0.0
        all_ious, all_recalls = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                all_ious.append(iou_per_class(preds.cpu(), lbls.cpu(), args.num_classes))
                all_recalls.append(recall_per_class(preds.cpu(), lbls.cpu(), args.num_classes))
        val_loss /= n_val
        mean_ious = np.nanmean(np.stack(all_ious), axis=0)
        mean_recalls = np.nanmean(np.stack(all_recalls), axis=0)
        target_iou = np.nanmean(mean_ious[1:])  # mean over class_10/12 only, ignoring background
        target_recalls = mean_recalls[1:]
        min_recall = np.nanmin(target_recalls)
        mean_recall = np.nanmean(target_recalls)
        score = {"iou": target_iou, "min_recall": min_recall, "mean_recall": mean_recall}[args.select_by]
        score_history.append(0.0 if np.isnan(score) else score)
        smoothed_score = float(np.mean(score_history[-args.smooth_window:]))

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs}  lr={current_lr:.2e}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"target_iou={target_iou:.3f} min_recall={min_recall:.3f} "
              f"{args.select_by}_smoothed={smoothed_score:.3f}  "
              f"IoU: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(class_names, mean_ious)) +
              "  Recall: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(class_names, mean_recalls)))

        if smoothed_score > best_score:
            best_score = smoothed_score
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": args.num_classes,
                "architecture": args.architecture,
                "encoder_name": args.encoder_name,
            }, out_path)
            print(f"  -> saved new best checkpoint to {out_path} ({args.select_by}_smoothed={smoothed_score:.3f})")

    print(f"Done. Best smoothed {args.select_by}={best_score:.3f}")


if __name__ == "__main__":
    main()
