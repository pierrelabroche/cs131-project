"""
Thin vs. thick vessel sensitivity analysis on the DRIVE training set.

For each method, computes sensitivity separately on thin vessels (radius <= 3px)
and thick vessels (radius > 3px) using a distance-transform caliber estimate.

Usage:
    python src/caliber_analysis.py

Output:
    - Prints a table of thin/thick sensitivity per method
    - Saves results/figures/caliber_analysis.png
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess
from canny import canny
from gabor import gabor_segment
from color_threshold import color_threshold_segment
from fusion import average_fusion_segment, weighted_fusion_segment
from evaluate import split_by_caliber

IMAGE_DIR = "data/DRIVE/training/images"
TRUTH_DIR = "data/DRIVE/training/1st_manual"
MASK_DIR  = "data/DRIVE/training/mask"
OUT_DIR   = "results/figures"
RADIUS_THRESHOLD = 1.5


def load_binary(path):
    img = io.imread(path)
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return (img > 0).astype(np.uint8)


def find_matching_file(folder, image_id):
    matches = glob.glob(os.path.join(folder, f"{image_id}*"))
    if not matches:
        raise FileNotFoundError(f"No match for {image_id} in {folder}")
    return matches[0]


def get_predictions(img, enhanced):
    """Return (binary_pred, method_name) for every method."""
    edges    = canny(enhanced, sigma=1.0, low=0.05, high=0.15)
    pred_canny = binary_fill_holes(closing(edges, disk(2))).astype(np.uint8)

    pred_gabor, _ = gabor_segment(enhanced)
    pred_color, _ = color_threshold_segment(img)
    pred_avg,   _ = average_fusion_segment(img, enhanced)
    pred_wt,    _ = weighted_fusion_segment(img, enhanced)

    return [
        (pred_canny, "Canny"),
        (pred_gabor, "Gabor"),
        (pred_color, "Color Threshold"),
        (pred_avg,   "Fusion (avg)"),
        (pred_wt,    "Fusion (weighted)"),
    ]


def sensitivity_on(pred, vessel_mask):
    """Sensitivity restricted to pixels in vessel_mask."""
    tp = np.sum((pred == 1) & (vessel_mask == 1))
    fn = np.sum((pred == 0) & (vessel_mask == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Download DRIVE first.")
        return

    method_names = ["Canny", "Gabor", "Color Threshold", "Fusion (avg)", "Fusion (weighted)"]
    thin_sens  = {m: [] for m in method_names}
    thick_sens = {m: [] for m in method_names}

    for i, image_path in enumerate(image_paths):
        filename  = os.path.basename(image_path)
        image_id  = filename.split("_")[0]
        gt_path   = find_matching_file(TRUTH_DIR, image_id)
        mask_path = find_matching_file(MASK_DIR, image_id)

        img         = io.imread(image_path)
        _, enhanced = preprocess(img)
        gt          = load_binary(gt_path)
        fov_mask    = load_binary(mask_path)

        thin_mask, thick_mask = split_by_caliber(gt, radius_threshold=RADIUS_THRESHOLD)

        # restrict caliber masks to FOV
        thin_mask  = thin_mask  & fov_mask
        thick_mask = thick_mask & fov_mask

        predictions = get_predictions(img, enhanced)
        for pred, name in predictions:
            thin_sens[name].append(sensitivity_on(pred, thin_mask))
            thick_sens[name].append(sensitivity_on(pred, thick_mask))

        print(f"[{i+1:2d}/{len(image_paths)}] {filename}")

    # Compute means
    mean_thin  = {m: np.mean(thin_sens[m])  for m in method_names}
    mean_thick = {m: np.mean(thick_sens[m]) for m in method_names}

    print(f"\n=== Thin vs Thick Vessel Sensitivity on DRIVE (radius threshold = {RADIUS_THRESHOLD}px) ===")
    print(f"{'Method':<22} {'Thin Sens':>10} {'Thick Sens':>11} {'Gap':>8}")
    print("-" * 55)
    for m in method_names:
        gap = mean_thick[m] - mean_thin[m]
        print(f"{m:<22} {mean_thin[m]:>10.4f} {mean_thick[m]:>11.4f} {gap:>8.4f}")

    # Bar chart
    os.makedirs(OUT_DIR, exist_ok=True)
    x = np.arange(len(method_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_thin  = ax.bar(x - width/2, [mean_thin[m]  for m in method_names], width,
                        label=f"Thin  (radius ≤ {RADIUS_THRESHOLD}px)", color="#e15759")
    bars_thick = ax.bar(x + width/2, [mean_thick[m] for m in method_names], width,
                        label=f"Thick (radius > {RADIUS_THRESHOLD}px)",  color="#4e79a7")

    ax.set_ylabel("Sensitivity (mean over 20 images)")
    ax.set_title("Thin vs. Thick Vessel Sensitivity by Method on DRIVE Set")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.bar_label(bars_thin,  fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars_thick, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "caliber_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
