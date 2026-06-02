import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

from canny import canny
from color_threshold import color_threshold_segment
from gabor import gabor_segment
from fusion import average_fusion_segment, max_fusion_segment, min_fusion_segment, weighted_fusion_segment
from evaluate import compute_metrics
from preprocessing import preprocess

IMAGE_DIR = "data/DRIVE/training/images"
TRUTH_DIR = "data/DRIVE/training/1st_manual"
MASK_DIR  = "data/DRIVE/training/mask"


def load_binary(path):
    """
    Load an image from disk and convert it to a binary mask (0 or 1).
    Handles grayscale, RGB, and multi-channel images.
    """
    img = io.imread(path)
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return (img > 0).astype(np.uint8)


def find_matching_file(folder, image_id):
    """
    Find a file in folder whose name starts with image_id.
    DRIVE filenames follow different conventions per split:
        image:  21_training.tif
        truth:  21_manual1.gif
        mask:   21_training_mask.gif
    """
    matches = glob.glob(os.path.join(folder, f"{image_id}*"))
    if not matches:
        raise FileNotFoundError(f"No matching file found for {image_id} in {folder}")
    return matches[0]


def run_canny(img, enhanced):
    """
    Run Canny edge detection and convert edges to filled vessel regions
    via morphological closing and hole filling.
    """
    edges = canny(enhanced, sigma=1.0, low=0.05, high=0.15)
    closed = closing(edges, disk(2))
    return binary_fill_holes(closed).astype(np.uint8)


def run_color_threshold(img, enhanced):
    """
    Run color thresholding segmentation (inverted L channel + Otsu).
    Operates on the original RGB image, not the preprocessed green channel.
    """
    binary, _ = color_threshold_segment(img)
    return binary


def run_gabor(img, enhanced):
    """
    Run Gabor filter bank segmentation (Soares et al. 2006, simplified).
    Returns the binary vessel mask; discards the raw response map.
    """
    binary, _ = gabor_segment(enhanced)
    return binary


def run_fusion(img, enhanced):
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
    "fusion_avg":       run_fusion,
    "fusion_max":       run_fusion_max,
    "fusion_min":       run_fusion_min,
    "fusion_weighted":  run_fusion_weighted,
}


def main():
    parser = argparse.ArgumentParser(description="Run a vessel segmentation method on the DRIVE training set.")
    parser.add_argument("--method", required=True, choices=METHODS.keys(),
                        help="Segmentation method to run")
    args = parser.parse_args()

    method_fn = METHODS[args.method]
    out_dir   = f"outputs/{args.method}/training"
    os.makedirs(out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
    results = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        image_id = filename.split("_")[0]

        gt_path   = find_matching_file(TRUTH_DIR, image_id)
        mask_path = find_matching_file(MASK_DIR, image_id)

        img             = io.imread(image_path)
        _, enhanced     = preprocess(img)
        gt              = load_binary(gt_path)
        fov_mask        = load_binary(mask_path)

        pred = method_fn(img, enhanced)

        metrics          = compute_metrics(pred, gt, fov_mask)
        metrics["image"] = filename
        results.append(metrics)

        out_path = os.path.join(out_dir, f"{image_id}_{args.method}_training.png")
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

    csv_path = os.path.join(out_dir, f"{args.method}_training_metrics.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved metrics to {csv_path}")
    print(f"Saved masks to {out_dir}")


if __name__ == "__main__":
    main()
