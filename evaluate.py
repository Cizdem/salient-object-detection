import os
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from data_loader import build_dataloaders
from sod_model   import build_model

def compute_metrics(
    preds:   torch.Tensor,
    targets: torch.Tensor,   
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> dict:
    
    bin_preds = (preds > threshold).float()

    p = bin_preds.view(-1)
    t = targets.view(-1)

    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()

    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = 2 * precision * recall / (precision + recall + smooth)
    mae       = (preds - targets).abs().mean().item()
    accuracy  = (tp + tn) / (tp + fp + fn + tn + smooth)

    return {
        "iou":       iou,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "mae":       mae,
        "accuracy":  accuracy,
    }


def threshold_analysis(
    preds: torch.Tensor,
    targets: torch.Tensor,
    thresholds=None,
) -> dict:
    
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    results = {t: compute_metrics(preds, targets, threshold=t) for t in thresholds}
    return results

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    save_dir: str = "results",
    n_visualise: int = 8,
):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()

    all_preds, all_targets, all_imgs, all_names = [], [], [], []

    for imgs, masks, names in tqdm(loader, desc="[Evaluate]"):
        imgs_d  = imgs.to(device, non_blocking=True)
        preds   = model(imgs_d).cpu()

        all_preds.append(preds)
        all_targets.append(masks)
        all_imgs.append(imgs)
        all_names.extend(names)

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_imgs    = torch.cat(all_imgs,    dim=0)

    metrics = compute_metrics(all_preds, all_targets)
    print("\n" + "─"*50)
    print("  TEST SET EVALUATION RESULTS")
    print("─"*50)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("─"*50)

    with open(Path(save_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    per_iou = []
    for i in range(len(all_preds)):
        m = compute_metrics(all_preds[i:i+1], all_targets[i:i+1])
        per_iou.append(m["iou"])

    indices = sorted(range(len(per_iou)), key=lambda i: per_iou[i], reverse=True)
    best    = indices[:n_visualise // 2]
    worst   = indices[-n_visualise // 2:]
    sample_idx = best + worst
    labels_txt = ["Best"] * len(best) + ["Worst"] * len(worst)

    save_prediction_grid(
        all_imgs, all_targets, all_preds, all_names,
        sample_idx, labels_txt,
        save_path=str(Path(save_dir) / "prediction_grid.png"),
    )
    
    th_results = threshold_analysis(all_preds, all_targets)
    plot_pr_curve(th_results, save_path=str(Path(save_dir) / "pr_curve.png"))

    print(f"\n[Evaluate] Results saved → '{save_dir}'")
    return metrics

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    img = (img_t * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _overlay(img_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = img_np.copy().astype(float)
    red_channel = np.zeros_like(img_np)
    red_channel[..., 0] = 255
    overlay = overlay * (1 - alpha * mask_np[..., None]) + \
              red_channel * alpha * mask_np[..., None]
    return overlay.astype(np.uint8)


def save_prediction_grid(
    imgs, gt_masks, pred_masks, names,
    indices, labels,
    save_path: str = "prediction_grid.png",
):
    
    n    = len(indices)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.5, n * 3.5))
    if n == 1:
        axes = [axes]

    col_titles = ["Input Image", "GT Mask", "Predicted Mask", "Overlay"]

    for row, (idx, lbl) in enumerate(zip(indices, labels)):
        img_np  = _denorm(imgs[idx])
        gt_np   = gt_masks[idx].squeeze().numpy()
        pred_np = (pred_masks[idx].squeeze().numpy() > 0.5).astype(float)
        ov_np   = _overlay(img_np, pred_np)

        m       = compute_metrics(pred_masks[idx:idx+1], gt_masks[idx:idx+1])

        for col, data in enumerate([img_np, gt_np, pred_np, ov_np]):
            ax = axes[row][col]
            if col in (1, 2):
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(data)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight="bold")

            if col == 0:
                short = names[idx][:18] if names else str(idx)
                ax.set_ylabel(
                    f"[{lbl}]\n{short}\nIoU={m['iou']:.3f}",
                    fontsize=8, rotation=0, labelpad=80, va="center"
                )
            ax.axis("off")

    plt.suptitle("SOD Predictions – Best & Worst Samples", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualise] Grid saved → {save_path}")


def plot_training_curves(log_csv: str, save_path: str = "training_curves.png"):
    
    epochs, train_loss, val_loss, iou_vals, f1_vals = [], [], [], [], []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            iou_vals.append(float(row["iou"]))
            f1_vals.append(float(row["f1"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="#FF5722", linewidth=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Metrics
    ax2.plot(epochs, iou_vals, label="Val IoU", color="#4CAF50", linewidth=2)
    ax2.plot(epochs, f1_vals,  label="Val F1",  color="#9C27B0", linewidth=2)
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves → {save_path}")


def plot_pr_curve(th_results: dict, save_path: str = "pr_curve.png"):
    precisions = [v["precision"] for v in th_results.values()]
    recalls    = [v["recall"]    for v in th_results.values()]
    f1s        = [v["f1"]        for v in th_results.values()]

    best_idx = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, "o-", color="#2196F3", linewidth=2, label="PR Curve")
    ax.scatter([recalls[best_idx]], [precisions[best_idx]],
               color="red", s=100, zorder=5,
               label=f"Best F1={f1s[best_idx]:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] PR curve → {save_path}")


def plot_metric_comparison(
    baseline_metrics: dict,
    improved_metrics: dict,
    save_path: str = "metric_comparison.png",
):
    
    keys   = ["iou", "precision", "recall", "f1", "accuracy"]
    base_v = [baseline_metrics.get(k, 0) for k in keys]
    impr_v = [improved_metrics.get(k, 0) for k in keys]

    x   = np.arange(len(keys))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - w/2, base_v, w, label="Baseline", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w/2, impr_v, w, label="Improved", color="#FF5722", alpha=0.85)

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score"); ax.set_title("Baseline vs Improved – Test Metrics")
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Comparison chart → {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SOD model on test set")
    parser.add_argument("--data_dir",    default="data/synthetic")
    parser.add_argument("--model_path",  required=True, help="Path to best_model.pth")
    parser.add_argument("--variant",     default="baseline", choices=["baseline","improved"])
    parser.add_argument("--image_size",  type=int, default=128)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--save_dir",    default="results")
    parser.add_argument("--log_csv",     default=None, help="Path to training_log.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.variant, input_size=args.image_size).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"[Evaluate] Loaded weights from {args.model_path}")

    loaders = build_dataloaders(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    metrics = evaluate_model(model, loaders["test"], device, save_dir=args.save_dir)

    if args.log_csv and os.path.exists(args.log_csv):
        plot_training_curves(args.log_csv,
                             save_path=str(Path(args.save_dir) / "training_curves.png"))
