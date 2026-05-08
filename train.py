import os
import csv
import time
import json
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_loader import build_dataloaders, generate_synthetic_dataset
from sod_model   import build_model, SODLoss
from evaluate    import compute_metrics

DEFAULT_CONFIG = {
    "data_dir":    "data/synthetic",
    "image_size":  128,
    "batch_size":  16,
    "num_workers": 2,
    "variant":     "baseline",
    "dropout":     0.2,
    "epochs":      25,
    "lr":          1e-3,
    "early_stop_patience": 5,
    "output_dir":  "runs/experiment",
    "bce_weight":  1.0,
    "iou_weight":  0.5,
}

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer, scheduler):
    if not os.path.exists(path):
        return 0

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    epoch = ckpt.get("epoch", 0)
    print(f"  [Checkpoint] Resumed from epoch {epoch}  (path={path})")
    return epoch

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = running_bce = running_iou = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for imgs, masks, _ in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds                   = model(imgs)
        loss, bce_v, iou_v      = criterion(preds, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_bce  += bce_v
        running_iou  += iou_v
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return running_loss / n, running_bce / n, running_iou / n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, masks, _ in tqdm(loader, desc="  [Val]", leave=False):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        preds = model(imgs)
        loss, _, _ = criterion(preds, masks)
        running_loss += loss.item()

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)
    return running_loss / len(loader), metrics

def train(config: dict):
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Variant : {config['variant']}")
    print(f"  Device  : {device}")
    print(f"  Out dir : {out_dir}")
    print(f"{'='*60}\n")
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    loaders = build_dataloaders(
        config["data_dir"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    
    model = build_model(
        config["variant"],
        input_size=config["image_size"],
        dropout=config.get("dropout", 0.2),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3,
                              factor=0.5)
    criterion = SODLoss(
        bce_weight=config["bce_weight"],
        iou_weight=config["iou_weight"],
    )

    ckpt_path  = str(out_dir / "checkpoint_latest.pth")
    start_epoch = load_checkpoint(ckpt_path, model, optimizer, scheduler)
    start_epoch += 1

    log_path = out_dir / "training_log.csv"
    log_file = open(log_path, "a", newline="")
    writer   = csv.writer(log_file)
    if start_epoch == 1:
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "iou", "precision", "recall", "f1", "lr", "time_s"])
        
    best_val_loss = float("inf")
    patience_ctr  = 0
    
    for epoch in range(start_epoch, config["epochs"] + 1):
        t0 = time.time()

        train_loss, _, _ = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
            epoch, config["epochs"]
        )

        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch:03d}/{config['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"IoU={val_metrics['iou']:.4f}  F1={val_metrics['f1']:.4f}  "
            f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_metrics['iou']:.6f}", f"{val_metrics['precision']:.6f}",
            f"{val_metrics['recall']:.6f}", f"{val_metrics['f1']:.6f}",
            f"{lr_now:.6e}", f"{elapsed:.1f}",
        ])
        log_file.flush()

        save_checkpoint(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss":        val_loss,
                "config":          config,
            },
            ckpt_path,
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path     = str(out_dir / "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  [Best] New best val_loss={best_val_loss:.4f} → {best_path}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            print(f"  [EarlyStop] No improvement ({patience_ctr}/{config['early_stop_patience']})")

        if patience_ctr >= config["early_stop_patience"]:
            print(f"\n[EarlyStop] Stopping at epoch {epoch}. "
                  f"Best val_loss={best_val_loss:.4f}")
            break

    log_file.close()
    print(f"\nTraining complete. Logs → {log_path}")
    print(f"Best model    → {out_dir / 'best_model.pth'}")
    return str(out_dir / "best_model.pth")

def run_experiments(data_dir: str = "data/synthetic"):
    results = {}
    for variant in ["baseline", "improved"]:
        cfg = {**DEFAULT_CONFIG,
               "variant":    variant,
               "data_dir":   data_dir,
               "output_dir": f"runs/{variant}"}
        best_path = train(cfg)
        results[variant] = best_path
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOD model")
    parser.add_argument("--data_dir",  default="data/synthetic", help="Dataset root")
    parser.add_argument("--variant",   default="baseline",       choices=["baseline","improved"])
    parser.add_argument("--epochs",    type=int,   default=25)
    parser.add_argument("--batch_size",type=int,   default=16)
    parser.add_argument("--image_size",type=int,   default=128)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--out_dir",   default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data before training")
    parser.add_argument("--run_all",   action="store_true",
                        help="Run both baseline and improved experiments")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(args.data_dir, n_samples=200, image_size=args.image_size)

    if args.run_all:
        run_experiments(args.data_dir)
    else:
        cfg = {
            **DEFAULT_CONFIG,
            "data_dir":   args.data_dir,
            "variant":    args.variant,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr":         args.lr,
            "output_dir": args.out_dir or f"runs/{args.variant}",
        }
        train(cfg)
