"""
demo_local.py
Simple local demo - picks a random image and shows saliency prediction.
Run: python demo_local.py --model_path runs/improved/best_model.pth
"""

import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms.functional as TF

from sod_model import build_model

# ── Settings ──────────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def load_model(model_path, variant="improved", image_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(variant, input_size=image_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded | device={device}")
    return model, device

@torch.no_grad()
def predict(model, device, img_path, image_size=128):
    pil     = Image.open(img_path).convert("RGB")
    resized = TF.resize(pil, [image_size, image_size])
    t       = (TF.to_tensor(resized) - _MEAN) / _STD

    t0   = time.perf_counter()
    pred = model(t.unsqueeze(0).to(device)).cpu().squeeze().numpy()
    ms   = (time.perf_counter() - t0) * 1000

    return np.array(resized), pred, ms

def show(img_np, prob, ms, img_name):
    binary = (prob > 0.5).astype(np.float32)
    heat   = (cm.jet(prob)[:, :, :3] * 255).astype(np.uint8)
    alpha  = 0.5
    overlay = (img_np * (1 - alpha * binary[..., None]) +
               np.array([255, 0, 0]) * alpha * binary[..., None]).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f"Salient Object Detection  |  {img_name}  |  Inference: {ms:.1f} ms",
                 fontsize=12, fontweight="bold")

    panels = [
        (img_np,  "Input Image",     None),
        (binary,  "Saliency Mask",   "gray"),
        (heat,    "Soft Heatmap",    None),
        (overlay, "Overlay",         None),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default="runs/improved/best_model.pth")
    parser.add_argument("--variant",     default="improved")
    parser.add_argument("--image_size",  type=int, default=128)
    parser.add_argument("--data_dir",    default="data/ECSSD")
    parser.add_argument("--image_path",  default=None,
                        help="Specific image to use (optional)")
    args = parser.parse_args()

    model, device = load_model(args.model_path, args.variant, args.image_size)

    # Pick image
    if args.image_path:
        img_path = Path(args.image_path)
    else:
        images   = list(Path(args.data_dir, "images").glob("*.jpg"))
        img_path = random.choice(images)
        print(f"Random image: {img_path.name}")

    img_np, prob, ms = predict(model, device, img_path, args.image_size)
    show(img_np, prob, ms, img_path.name)
