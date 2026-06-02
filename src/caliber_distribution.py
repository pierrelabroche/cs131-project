"""
Analyse the distribution of vessel radii across all 20 DRIVE training GT masks.
Uses distance transform to estimate local vessel radius at each vessel pixel.

Output:
    - Prints pixel counts and split fractions for radius thresholds 1–8px
    - Saves results/figures/caliber_distribution.png
      (histogram + cumulative distribution + per-threshold split table)

Usage:
    python src/caliber_distribution.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(__file__))

IMAGE_DIR = "data/DRIVE/training/images"
TRUTH_DIR = "data/DRIVE/training/1st_manual"
MASK_DIR  = "data/DRIVE/training/mask"
OUT_DIR   = "results/figures"


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
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Download DRIVE first.")
        return

    all_radii = []

    for image_path in image_paths:
        filename  = os.path.basename(image_path)
        image_id  = filename.split("_")[0]
        gt_path   = find_matching_file(TRUTH_DIR, image_id)
        mask_path = find_matching_file(MASK_DIR, image_id)

        gt       = load_binary(gt_path)
        fov_mask = load_binary(mask_path)

        dist = distance_transform_edt(gt > 0)
        # collect radii of vessel pixels inside FOV only
        vessel_radii = dist[(gt > 0) & (fov_mask > 0)]
        all_radii.append(vessel_radii)

    all_radii = np.concatenate(all_radii)
    total = len(all_radii)

    # --- print split table ---
    print(f"Total vessel pixels (across 20 images): {total:,}")
    print(f"\n{'Threshold':>10} {'Thin px':>10} {'Thick px':>10} {'Thin %':>8} {'Thick %':>8}")
    print("-" * 52)
    for t in [1, 1.5, 2, 3]:
        n_thin  = np.sum(all_radii <= t)
        n_thick = np.sum(all_radii >  t)
        print(f"{t:>10.1f}  {n_thin:>10,}  {n_thick:>10,}  {100*n_thin/total:>7.1f}%  {100*n_thick/total:>7.1f}%")

    # --- plot ---
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of radii
    ax = axes[0]
    max_r = int(np.ceil(all_radii.max()))
    bins  = np.arange(0.5, max_r + 1.5, 1)
    ax.hist(all_radii, bins=bins, color="#4e79a7", edgecolor="white", linewidth=0.4)
    for t in [1, 1.5, 2, 3]:
        ax.axvline(t, linestyle="--", linewidth=1.2,
                   label=f"t={t}  ({100*np.mean(all_radii<=t):.0f}% thin)")
    ax.set_xlabel("Vessel radius (px)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Vessel Radius Distribution\n(all 20 DRIVE training GT masks)")
    ax.legend(fontsize=9)

    # Right: cumulative fraction vs threshold
    ax2 = axes[1]
    thresholds = np.linspace(0, max_r, 300)
    cumfrac = [np.mean(all_radii <= t) for t in thresholds]
    ax2.plot(thresholds, cumfrac, color="#4e79a7", linewidth=2)
    for t in [1, 1.5, 2, 3]:
        frac = np.mean(all_radii <= t)
        ax2.axvline(t, linestyle="--", linewidth=1.2, label=f"t={t} → {100*frac:.0f}% thin")
        ax2.plot(t, frac, "o", color="black", markersize=5)
    ax2.set_xlabel("Radius threshold (px)")
    ax2.set_ylabel("Fraction of vessel pixels labelled 'thin'")
    ax2.set_title("Cumulative Fraction of Vessel Pixels\nvs. Radius Threshold")
    ax2.set_xlim(0, max_r)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "caliber_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
