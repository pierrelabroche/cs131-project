"""
Thin vs. thick vessel sensitivity analysis on DRIVE or STARE.

For each method, computes sensitivity separately on thin vessels (radius <= 1.5px)
and thick vessels (radius > 1.5px) using a distance-transform caliber estimate.

Usage:
    python src/caliber_analysis.py --dataset drive
    python src/caliber_analysis.py --dataset stare

Output:
    - Prints a table of thin/thick sensitivity per method
    - Saves results/figures/caliber_analysis_<dataset>.png
"""

import os
import sys
import gzip
import io
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess
from canny import canny
from gabor import gabor_segment
from color_threshold import color_threshold_segment
from fusion import weighted_fusion_segment
from evaluate import split_by_caliber

RADIUS_THRESHOLD = 1.5
OUT_DIR = "results/figures"

DATASET_CONFIG = {
    "drive": {
        "image_dir":  "data/DRIVE/training/images",
        "truth_dir":  "data/DRIVE/training/1st_manual",
        "mask_dir":   "data/DRIVE/training/mask",
        "pattern":    "*.tif",
        "n_images":   20,
    },
    "stare": {
        "image_dir":  "data/STARE/stare-images",
        "truth_dir":  "data/STARE/labels-ah",
        "mask_dir":   None,
        "pattern":    "*.ppm.gz",
        "n_images":   20,
    },
}


# --- loaders ---

def load_binary_drive(path):
    img = skio.imread(path)
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return (img > 0).astype(np.uint8)


def load_ppm_gz(path):
    with gzip.open(path, "rb") as f:
        return np.array(Image.open(io.BytesIO(f.read())))


def load_binary_stare(path):
    img = load_ppm_gz(path)
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return (img > 0).astype(np.uint8)


def brightness_fov_mask(img_rgb, threshold=50):
    return (img_rgb.max(axis=2) > threshold).astype(np.uint8)


def find_matching_file(folder, image_id):
    matches = glob.glob(os.path.join(folder, f"{image_id}*"))
    if not matches:
        raise FileNotFoundError(f"No match for {image_id} in {folder}")
    return matches[0]


# --- predictions ---

def get_predictions(img, enhanced):
    edges      = canny(enhanced, sigma=1.0, low=0.05, high=0.15)
    pred_canny = binary_fill_holes(closing(edges, disk(2))).astype(np.uint8)
    pred_gabor, _ = gabor_segment(enhanced)
    pred_color, _ = color_threshold_segment(img)
    pred_wt,    _ = weighted_fusion_segment(img, enhanced)
    return [
        (pred_canny, "Canny"),
        (pred_gabor, "Gabor"),
        (pred_color, "Color Threshold"),
        (pred_wt,    "Fusion (weighted)"),
    ]


def sensitivity_on(pred, vessel_mask):
    tp = np.sum((pred == 1) & (vessel_mask == 1))
    fn = np.sum((pred == 0) & (vessel_mask == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


# --- main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["drive", "stare"], default="drive")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    is_stare = args.dataset == "stare"

    image_paths = sorted(glob.glob(os.path.join(cfg["image_dir"], cfg["pattern"])))
    if not image_paths:
        print(f"No images found in {cfg['image_dir']}.")
        return

    method_names = ["Canny", "Gabor", "Color Threshold", "Fusion (weighted)"]
    thin_sens  = {m: [] for m in method_names}
    thick_sens = {m: [] for m in method_names}

    for i, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)

        if is_stare:
            image_id = filename.replace(".ppm.gz", "")
            gt_path  = os.path.join(cfg["truth_dir"], image_id + ".ah.ppm.gz")
            if not os.path.exists(gt_path):
                print(f"Skipping {filename} — no label found")
                continue
            img      = load_ppm_gz(image_path)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            gt       = load_binary_stare(gt_path)
            fov_mask = brightness_fov_mask(img)
        else:
            image_id = filename.split("_")[0]
            gt_path   = find_matching_file(cfg["truth_dir"], image_id)
            mask_path = find_matching_file(cfg["mask_dir"],  image_id)
            img      = skio.imread(image_path)
            gt       = load_binary_drive(gt_path)
            fov_mask = load_binary_drive(mask_path)

        _, enhanced = preprocess(img)

        thin_mask, thick_mask = split_by_caliber(gt, radius_threshold=RADIUS_THRESHOLD)
        thin_mask  = thin_mask  & fov_mask
        thick_mask = thick_mask & fov_mask

        predictions = get_predictions(img, enhanced)
        for pred, name in predictions:
            thin_sens[name].append(sensitivity_on(pred, thin_mask))
            thick_sens[name].append(sensitivity_on(pred, thick_mask))

        print(f"[{i+1:2d}/{len(image_paths)}] {filename}")

    mean_thin  = {m: np.mean(thin_sens[m])  for m in method_names}
    mean_thick = {m: np.mean(thick_sens[m]) for m in method_names}

    dataset_label = args.dataset.upper()
    print(f"\n=== Thin vs Thick Vessel Sensitivity on {dataset_label} (radius threshold = {RADIUS_THRESHOLD}px) ===")
    print(f"{'Method':<22} {'Thin Sens':>10} {'Thick Sens':>11} {'Gap':>8}")
    print("-" * 55)
    for m in method_names:
        gap = mean_thick[m] - mean_thin[m]
        print(f"{m:<22} {mean_thin[m]:>10.4f} {mean_thick[m]:>11.4f} {gap:>8.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    x = np.arange(len(method_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_thin  = ax.bar(x - width/2, [mean_thin[m]  for m in method_names], width,
                        label=f"Thin  (radius ≤ {RADIUS_THRESHOLD}px)", color="#e15759")
    bars_thick = ax.bar(x + width/2, [mean_thick[m] for m in method_names], width,
                        label=f"Thick (radius > {RADIUS_THRESHOLD}px)",  color="#4e79a7")

    ax.set_ylabel("Sensitivity (mean over 20 images)")
    ax.set_title(f"Thin vs. Thick Vessel Sensitivity by Method — {dataset_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.bar_label(bars_thin,  fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars_thick, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"caliber_analysis_{args.dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
