"""
Generate ROC curves for all methods on DRIVE or STARE.

For each method, compute a per-image ROC curve, then average TPR and FPR
across all 20 images at shared FPR grid points. AUC reported as mean ± std
of per-image AUCs.

Usage:
    python src/plot_roc.py --dataset drive
    python src/plot_roc.py --dataset stare

Output:
    results/figures/roc_curves_<dataset>.png
    outputs/roc_auc_per_image_<dataset>.csv
"""

import os
import sys
import gzip
import io
import glob
import argparse
import numpy as np
import pandas as pd
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
from evaluate import roc_curve, auc

OUT_DIR      = "results/figures"
N_THRESHOLDS = 200
FPR_GRID     = np.linspace(0, 1, 500)

DATASET_CONFIG = {
    "drive": {
        "image_dir": "data/DRIVE/training/images",
        "truth_dir": "data/DRIVE/training/1st_manual",
        "mask_dir":  "data/DRIVE/training/mask",
        "pattern":   "*.tif",
        "label":     "DRIVE Training Set",
    },
    "stare": {
        "image_dir": "data/STARE/stare-images",
        "truth_dir": "data/STARE/labels-ah",
        "mask_dir":  None,
        "pattern":   "*.ppm.gz",
        "label":     "STARE Dataset",
    },
}

COLORS = {
    "Canny":             "#e15759",
    "Gabor":             "#4e79a7",
    "Color Threshold":   "#59a14f",
    "Fusion (weighted)": "#b07aa1",
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


# --- response maps ---

def get_response_maps(img, enhanced):
    _, canny_response = canny(enhanced, sigma=1.0, low=0.05, high=0.15, return_response=True)
    _, gabor_response = gabor_segment(enhanced)
    _, L_enhanced     = color_threshold_segment(img)
    color_response    = L_enhanced.astype(np.float32) / 255.0
    _, fusion_response = weighted_fusion_segment(img, enhanced)
    return [
        (canny_response,  "Canny"),
        (gabor_response,  "Gabor"),
        (color_response,  "Color Threshold"),
        (fusion_response, "Fusion (weighted)"),
    ]


# --- averaging ---

def average_roc(per_image_tpr, per_image_fpr):
    tpr_interp = []
    for tpr, fpr in zip(per_image_tpr, per_image_fpr):
        order = np.argsort(fpr)
        tpr_interp.append(np.interp(FPR_GRID, fpr[order], tpr[order]))
    tpr_mat = np.stack(tpr_interp)
    return tpr_mat.mean(axis=0), tpr_mat.std(axis=0)


# --- main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["drive", "stare"], default="drive")
    args = parser.parse_args()

    cfg      = DATASET_CONFIG[args.dataset]
    is_stare = args.dataset == "stare"

    image_paths = sorted(glob.glob(os.path.join(cfg["image_dir"], cfg["pattern"])))
    if not image_paths:
        print(f"No images found in {cfg['image_dir']}.")
        return

    method_names    = ["Canny", "Gabor", "Color Threshold", "Fusion (weighted)"]
    per_image_tprs  = {m: [] for m in method_names}
    per_image_fprs  = {m: [] for m in method_names}
    per_image_aucs  = {m: [] for m in method_names}

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
            image_id  = filename.split("_")[0]
            gt_path   = find_matching_file(cfg["truth_dir"], image_id)
            mask_path = find_matching_file(cfg["mask_dir"],  image_id)
            img       = skio.imread(image_path)
            gt        = load_binary_drive(gt_path)
            fov_mask  = load_binary_drive(mask_path)

        _, enhanced = preprocess(img)

        responses = get_response_maps(img, enhanced)
        for response_map, name in responses:
            tpr, fpr = roc_curve(response_map, gt, fov_mask, n_thresholds=N_THRESHOLDS)
            per_image_tprs[name].append(tpr)
            per_image_fprs[name].append(fpr)
            per_image_aucs[name].append(auc(fpr, tpr))

        print(f"[{i+1:2d}/{len(image_paths)}] {filename}")

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    for name in method_names:
        mean_tpr, std_tpr = average_roc(per_image_tprs[name], per_image_fprs[name])
        mean_auc = np.mean(per_image_aucs[name])
        std_auc  = np.std(per_image_aucs[name])
        color    = COLORS[name]

        ax.plot(FPR_GRID, mean_tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {mean_auc:.4f} ± {std_auc:.4f})")
        ax.fill_between(FPR_GRID,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        color=color, alpha=0.15)

    ax.set_xlabel("FPR  (1 − Specificity)")
    ax.set_ylabel("TPR  (Sensitivity)")
    ax.set_title(f"ROC Curves — {cfg['label']}\n(mean ± 1 std across 20 images)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"roc_curves_{args.dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # Save per-image AUCs
    auc_rows = []
    for name in method_names:
        for img_path, auc_val in zip(image_paths, per_image_aucs[name]):
            auc_rows.append({"method": name, "image": os.path.basename(img_path), "auc": auc_val})
    auc_df  = pd.DataFrame(auc_rows)
    auc_csv = os.path.join("outputs", f"roc_auc_per_image_{args.dataset}.csv")
    os.makedirs("outputs", exist_ok=True)
    auc_df.to_csv(auc_csv, index=False)
    print(f"Saved {auc_csv}")


if __name__ == "__main__":
    main()
