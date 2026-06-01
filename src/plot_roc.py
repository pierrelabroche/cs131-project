"""
Generate ROC curves for all three methods on the DRIVE training set.

For each method, compute a per-image ROC curve, then average TPR and FPR
across all 20 images at shared FPR grid points.

Output: results/figures/roc_curves.png
"""

import os
import sys
import glob
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
from evaluate import roc_curve, auc

IMAGE_DIR = "data/DRIVE/training/images"
TRUTH_DIR = "data/DRIVE/training/1st_manual"
MASK_DIR  = "data/DRIVE/training/mask"
OUT_DIR   = "results/figures"

N_THRESHOLDS = 200
# Shared FPR grid for averaging per-image curves
FPR_GRID = np.linspace(0, 1, 500)


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


def get_response_maps(img, enhanced):
    """Return (response_map, label) for each method."""
    # Canny: normalized NMS magnitude before hysteresis
    _, canny_response = canny(enhanced, sigma=1.0, low=0.05, high=0.15, return_response=True)

    # Gabor: normalized filter-bank response
    _, gabor_response = gabor_segment(enhanced)

    # Color threshold: normalized inverted-L channel
    _, L_enhanced = color_threshold_segment(img)
    color_response = L_enhanced.astype(np.float32) / 255.0

    return [
        (canny_response, "Canny"),
        (gabor_response, "Gabor"),
        (color_response, "Color Threshold"),
    ]


def average_roc(per_image_tpr, per_image_fpr):
    """
    Interpolate each per-image curve onto a shared FPR grid, then average.
    Returns (mean_tpr, std_tpr) on FPR_GRID.
    """
    tpr_interp = []
    for tpr, fpr in zip(per_image_tpr, per_image_fpr):
        order = np.argsort(fpr)
        tpr_interp.append(np.interp(FPR_GRID, fpr[order], tpr[order]))
    tpr_mat = np.stack(tpr_interp)
    return tpr_mat.mean(axis=0), tpr_mat.std(axis=0)


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Download DRIVE first.")
        return

    # Accumulate per-image (tpr, fpr) for each method
    method_names = ["Canny", "Gabor", "Color Threshold"]
    per_image_tprs = {m: [] for m in method_names}
    per_image_fprs = {m: [] for m in method_names}

    for i, image_path in enumerate(image_paths):
        filename  = os.path.basename(image_path)
        image_id  = filename.split("_")[0]
        gt_path   = find_matching_file(TRUTH_DIR, image_id)
        mask_path = find_matching_file(MASK_DIR, image_id)

        img      = io.imread(image_path)
        _, enhanced = preprocess(img)
        gt       = load_binary(gt_path)
        fov_mask = load_binary(mask_path)

        responses = get_response_maps(img, enhanced)
        for response_map, name in responses:
            tpr, fpr = roc_curve(response_map, gt, fov_mask, n_thresholds=N_THRESHOLDS)
            per_image_tprs[name].append(tpr)
            per_image_fprs[name].append(fpr)

        print(f"[{i+1:2d}/{len(image_paths)}] {filename}")

    os.makedirs(OUT_DIR, exist_ok=True)

    colors = {"Canny": "#e15759", "Gabor": "#4e79a7", "Color Threshold": "#59a14f"}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    for name in method_names:
        mean_tpr, std_tpr = average_roc(per_image_tprs[name], per_image_fprs[name])
        area = auc(FPR_GRID, mean_tpr)
        color = colors[name]

        ax.plot(FPR_GRID, mean_tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {area:.4f})")
        ax.fill_between(FPR_GRID,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        color=color, alpha=0.15)

    ax.set_xlabel("FPR  (1 − Specificity)")
    ax.set_ylabel("TPR  (Sensitivity)")
    ax.set_title("ROC Curves — DRIVE Training Set\n(mean ± 1 std across 20 images)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "roc_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
