"""
Find the optimal binarization threshold for a fusion method by sweeping
thresholds over all 20 DRIVE training images and maximizing mean F1.

Usage:
    python src/tune_threshold.py --method fusion_avg
    python src/tune_threshold.py --method fusion_max
    python src/tune_threshold.py --method fusion_min

Output:
    - Prints the optimal threshold and its mean F1/sensitivity/specificity
    - Saves results/figures/threshold_tuning_<method>.png
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
from fusion import average_fusion_segment, max_fusion_segment, min_fusion_segment, weighted_fusion_segment
from evaluate import compute_metrics

IMAGE_DIR = "data/DRIVE/training/images"
TRUTH_DIR = "data/DRIVE/training/1st_manual"
MASK_DIR  = "data/DRIVE/training/mask"
OUT_DIR   = "results/figures"
N_THRESHOLDS = 100

FUSION_METHODS = {
    "fusion_avg":      average_fusion_segment,
    "fusion_max":      max_fusion_segment,
    "fusion_min":      min_fusion_segment,
    "fusion_weighted": weighted_fusion_segment,
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=FUSION_METHODS.keys())
    args = parser.parse_args()

    fusion_fn = FUSION_METHODS[args.method]
    thresholds = np.linspace(0, 1, N_THRESHOLDS)

    # per-threshold accumulators across images
    f1_per_thresh      = np.zeros(N_THRESHOLDS)
    sens_per_thresh    = np.zeros(N_THRESHOLDS)
    spec_per_thresh    = np.zeros(N_THRESHOLDS)

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Download DRIVE first.")
        return

    for i, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)
        image_id = filename.split("_")[0]
        gt_path   = find_matching_file(TRUTH_DIR, image_id)
        mask_path = find_matching_file(MASK_DIR, image_id)

        img         = io.imread(image_path)
        _, enhanced = preprocess(img)
        gt          = load_binary(gt_path)
        fov_mask    = load_binary(mask_path)

        # get the continuous fused map (ignore the binary mask)
        _, fused_map = fusion_fn(img, enhanced)

        for j, t in enumerate(thresholds):
            pred = (fused_map >= t).astype(np.uint8)
            m = compute_metrics(pred, gt, fov_mask)
            f1_per_thresh[j]   += m["f1"]
            sens_per_thresh[j] += m["sensitivity"]
            spec_per_thresh[j] += m["specificity"]

        print(f"[{i+1:2d}/{len(image_paths)}] {filename}")

    n = len(image_paths)
    f1_per_thresh   /= n
    sens_per_thresh /= n
    spec_per_thresh /= n

    best_idx = np.argmax(f1_per_thresh)
    best_t   = thresholds[best_idx]

    print(f"\n=== Optimal threshold for {args.method} ===")
    print(f"  Threshold:   {best_t:.3f}")
    print(f"  Mean F1:     {f1_per_thresh[best_idx]:.4f}")
    print(f"  Mean Sens:   {sens_per_thresh[best_idx]:.4f}")
    print(f"  Mean Spec:   {spec_per_thresh[best_idx]:.4f}")

    # plot
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_per_thresh,   label="F1",          color="#4e79a7", linewidth=2)
    ax.plot(thresholds, sens_per_thresh, label="Sensitivity",  color="#e15759", linewidth=1.5, linestyle="--")
    ax.plot(thresholds, spec_per_thresh, label="Specificity",  color="#59a14f", linewidth=1.5, linestyle="--")
    ax.axvline(best_t, color="black", linestyle=":", linewidth=1.5,
               label=f"Optimal t={best_t:.3f}  (F1={f1_per_thresh[best_idx]:.4f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score (mean over 20 images)")
    ax.set_title(f"Threshold Tuning — {args.method}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"threshold_tuning_{args.method}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
