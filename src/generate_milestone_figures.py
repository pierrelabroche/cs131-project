import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from canny import canny
from preprocessing import preprocess

IMAGE_PATH = "data/DRIVE/training/images/21_training.tif"
GT_PATH    = "data/DRIVE/training/1st_manual/21_manual1.gif"
MASK_PATH  = "data/DRIVE/training/mask/21_training_mask.gif"
OUT_DIR    = "results/figures"

def load():
    image = np.array(Image.open(IMAGE_PATH))
    if image.shape[2] == 4:
        image = image[:, :, :3]
    gt   = np.array(Image.open(GT_PATH))
    mask = np.array(Image.open(MASK_PATH))
    return image, (gt > 0).astype(np.uint8), (mask > 0).astype(np.uint8)

def run_canny(image):
    _, green = preprocess(image)
    edges  = canny(green, sigma=1.0, low=0.05, high=0.15)
    closed = closing(edges, disk(2))
    pred   = binary_fill_holes(closed).astype(np.uint8)
    return edges, pred

def figure2(image, edges):
    overlay = image.copy()
    overlay[edges == 1] = [255, 255, 0]  # yellow edges on original

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny Edges")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure2_canny.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure2_canny.png")

def figure3(image, pred, gt, fov_mask):
    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    inside = fov_mask == 1

    vis[inside & (pred == 1) & (gt == 1)] = [0,   200, 0  ]  # TP green
    vis[inside & (pred == 0) & (gt == 1)] = [220, 0,   0  ]  # FN red
    vis[inside & (pred == 1) & (gt == 0)] = [0,   0,   220]  # FP blue
    vis[inside & (pred == 0) & (gt == 0)] = [30,  30,  30 ]  # TN dark gray

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(vis)
    axes[2].set_title("Prediction  (green=TP  red=FN  blue=FP)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure3_mask_vs_gt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure3_mask_vs_gt.png")

if __name__ == "__main__":
    image, gt, fov_mask = load()
    edges, pred = run_canny(image)
    figure2(image, edges)
    figure3(image, pred, gt, fov_mask)
