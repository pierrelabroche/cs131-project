"""
Run a vessel segmentation method on all 20 STARE images.

STARE has no FOV mask files — a brightness-based mask is generated automatically:
pixels where max(R,G,B) <= 10 are outside the FOV (pure black background).

Usage:
    python src/run_stare.py --method canny
    python src/run_stare.py --method gabor
    python src/run_stare.py --method color_threshold
    python src/run_stare.py --method fusion_avg
    python src/run_stare.py --method fusion_weighted
    python src/run_stare.py --method fusion_max
    python src/run_stare.py --method fusion_min
    python src/run_stare.py --method canny --annotator vk

Outputs go to outputs/<method>/stare/ (PNG masks + <method>_stare_metrics.csv).
"""

import os
import sys
import glob
import gzip
import io
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess
from canny import canny
from gabor import gabor_segment
from color_threshold import color_threshold_segment
from fusion import average_fusion_segment, max_fusion_segment, min_fusion_segment, weighted_fusion_segment
from evaluate import compute_metrics

IMAGE_DIR  = "data/STARE/stare-images"
LABEL_DIR  = {"ah": "data/STARE/labels-ah", "vk": "data/STARE/labels-vk"}
LABEL_SUFFIX = {"ah": ".ah.ppm.gz", "vk": ".vk.ppm.gz"}


def load_ppm_gz(path):
    """Load a gzipped PPM file and return a numpy array."""
    with gzip.open(path, "rb") as f:
        return np.array(Image.open(io.BytesIO(f.read())))


def load_binary_gz(path):
    """Load a gzipped PPM label and return a binary (0/1) mask."""
    img = load_ppm_gz(path)
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return (img > 0).astype(np.uint8)


def brightness_fov_mask(img_rgb, threshold=50):
    """Generate FOV mask by thresholding max(R,G,B) channel."""
    return (img_rgb.max(axis=2) > threshold).astype(np.uint8)


def run_canny(img, enhanced):
    edges = canny(enhanced, sigma=1.0, low=0.05, high=0.15)
    return binary_fill_holes(closing(edges, disk(2))).astype(np.uint8)


def run_color_threshold(img, enhanced):
    binary, _ = color_threshold_segment(img)
    return binary


def run_gabor(img, enhanced):
    binary, _ = gabor_segment(enhanced)
    return binary


def run_fusion_avg(img, enhanced):
    binary, _ = average_fusion_segment(img, enhanced)
    return binary


def run_fusion_max(img, enhanced):
    binary, _ = max_fusion_segment(img, enhanced)
    return binary


def run_fusion_min(img, enhanced):
    binary, _ = min_fusion_segment(img, enhanced)
    return binary


def run_fusion_weighted(img, enhanced):
    binary, _ = weighted_fusion_segment(img, enhanced)
    return binary


METHODS = {
    "canny":            run_canny,
    "color_threshold":  run_color_threshold,
    "gabor":            run_gabor,
    "fusion_avg":       run_fusion_avg,
    "fusion_max":       run_fusion_max,
    "fusion_min":       run_fusion_min,
    "fusion_weighted":  run_fusion_weighted,
}


def main():
    parser = argparse.ArgumentParser(description="Run vessel segmentation on STARE dataset.")
    parser.add_argument("--method", required=True, choices=METHODS.keys())
    parser.add_argument("--annotator", choices=["ah", "vk"], default="ah",
                        help="Ground truth annotator to use (default: ah)")
    args = parser.parse_args()

    method_fn  = METHODS[args.method]
    label_dir  = LABEL_DIR[args.annotator]
    label_suf  = LABEL_SUFFIX[args.annotator]
    out_dir    = f"outputs/stare/{args.method}"
    os.makedirs(out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.ppm.gz")))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}.")
        return

    results = []

    for image_path in image_paths:
        filename  = os.path.basename(image_path)
        image_id  = filename.replace(".ppm.gz", "")
        gt_path   = os.path.join(label_dir, image_id + label_suf)

        if not os.path.exists(gt_path):
            print(f"Skipping {filename} — no label found at {gt_path}")
            continue

        img         = load_ppm_gz(image_path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        _, enhanced = preprocess(img)
        gt          = load_binary_gz(gt_path)
        fov_mask    = brightness_fov_mask(img)

        pred = method_fn(img, enhanced)

        metrics          = compute_metrics(pred, gt, fov_mask)
        metrics["image"] = filename
        results.append(metrics)

        out_path = os.path.join(out_dir, f"{image_id}_{args.method}.png")
        plt.imsave(out_path, pred, cmap="gray")

        print(
            f"{filename}: "
            f"Sens={metrics['sensitivity']:.4f}, "
            f"Spec={metrics['specificity']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    df = pd.DataFrame(results)
    cols = ["image", "sensitivity", "specificity", "accuracy", "f1", "TP", "TN", "FP", "FN"]
    df = df[cols]

    print("\n=== Per-image results ===")
    print(df.to_string(index=False))

    print("\n=== Mean results ===")
    print(df[["sensitivity", "specificity", "accuracy", "f1"]].mean())

    csv_path = os.path.join(out_dir, f"{args.method}_stare_metrics.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved metrics to {csv_path}")
    print(f"Saved masks to {out_dir}")


if __name__ == "__main__":
    main()
