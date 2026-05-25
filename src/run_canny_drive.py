import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

from canny import canny
from evaluate import compute_metrics
from preprocessing import preprocess
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

IMAGE_DIR = "data/DRIVE/training/images"
TRUTH_DIR = "data/DRIVE/training/1st_manual"
MASK_DIR = "data/DRIVE/training/mask"
OUT_DIR = "outputs/canny/training"

os.makedirs(OUT_DIR, exist_ok=True)

def load_binary(path):
    """
    Convert an image into a binary image
    """
    img = io.imread(path)
    img = np.squeeze(img)

    if img.ndim == 3:
        img = img[:, :, 0]

    return (img > 0).astype(np.uint8)


def find_matching_file(folder, image_id):
    """
    DRIVE filenames can be slightly different:
    image: 21_training.tif
    truth:    21_manual1.gif
    mask:  21_training_mask.gif

    This function finds files starting with the same image id.
    """
    matches = glob.glob(os.path.join(folder, f"{image_id}*"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No matching file found for {image_id} in {folder}")

    return matches[0]


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))

    results = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        image_id = filename.split("_")[0]

        # Load ground truth, FOV mask, and image
        groundtruth_path = find_matching_file(TRUTH_DIR, image_id)
        mask_path = find_matching_file(MASK_DIR, image_id)
        img = io.imread(image_path)

        _, green = preprocess(img)

        # Binary ground truth and mask
        gt = load_binary(groundtruth_path)
        fov_mask = load_binary(mask_path)

        # Run canny edge detector
        edges = canny(green, sigma=1.0, low=0.05, high=0.15)

        closed = closing(edges, disk(2))
        pred = binary_fill_holes(closed).astype(np.uint8)

        # Compute metrics
        metrics = compute_metrics(pred, gt, fov_mask)
        metrics["image"] = filename
        results.append(metrics)

        # Save each Canny output image
        out_path = os.path.join(OUT_DIR, f"{image_id}_canny_training.png")
        plt.imsave(out_path, pred, cmap="gray")

        print(
            f"{filename}: "
            f"Sens={metrics['sensitivity']:.4f}, "
            f"Spec={metrics['specificity']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    df = pd.DataFrame(results)

    cols = [
        "image",
        "sensitivity",
        "specificity",
        "accuracy",
        "f1",
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    df = df[cols]

    print("\n=== Per-image results ===")
    print(df.to_string(index=False))

    print("\n=== Mean results ===")
    print(df[["sensitivity", "specificity", "accuracy", "f1"]].mean())

    csv_path = os.path.join(OUT_DIR, "canny_training_metrics.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved metrics to {csv_path}")
    print(f"Saved Canny masks to {OUT_DIR}")


if __name__ == "__main__":
    main()