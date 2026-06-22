import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from canny import canny
from gabor import gabor_segment
from color_threshold import color_threshold_segment
from fusion import weighted_fusion_segment
from preprocessing import preprocess

IMAGE_PATH = "data/DRIVE/training/images/21_training.tif"
GT_PATH    = "data/DRIVE/training/1st_manual/21_manual1.gif"
MASK_PATH  = "data/DRIVE/training/mask/21_training_mask.gif"
OUT_DIR    = "results/figures"


def load():
    image = np.array(Image.open(IMAGE_PATH))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    gt   = np.array(Image.open(GT_PATH))
    mask = np.array(Image.open(MASK_PATH))
    return image, (gt > 0).astype(np.uint8), (mask > 0).astype(np.uint8)


def run_canny(image):
    _, enhanced = preprocess(image)
    edges  = canny(enhanced, sigma=1.0, low=0.05, high=0.15)
    closed = closing(edges, disk(2))
    pred   = binary_fill_holes(closed).astype(np.uint8)
    return edges, pred


def run_gabor(image):
    _, enhanced = preprocess(image)
    pred, _ = gabor_segment(enhanced)
    return pred


def run_color(image):
    pred, _ = color_threshold_segment(image)
    return pred


def run_fusion_weighted(image):
    _, enhanced = preprocess(image)
    pred, _ = weighted_fusion_segment(image, enhanced)
    return pred


def error_map(pred, gt, fov_mask):
    """
    Build RGB error map: TP=green, FN=red, FP=blue, TN=dark gray.
    """
    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    inside = fov_mask == 1
    vis[inside & (pred == 1) & (gt == 1)] = [0,   200, 0  ]
    vis[inside & (pred == 0) & (gt == 1)] = [220, 0,   0  ]
    vis[inside & (pred == 1) & (gt == 0)] = [0,   0,   220]
    vis[inside & (pred == 0) & (gt == 0)] = [30,  30,  30 ]
    return vis


def figure2(image, edges):
    """
    Figure 2: Original | Canny edges | Yellow overlay.
    """
    overlay = image.copy()
    overlay[edges == 1] = [255, 255, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image);        axes[0].set_title("Original")
    axes[1].imshow(edges, cmap="gray"); axes[1].set_title("Canny Edges")
    axes[2].imshow(overlay);      axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure2_canny.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure2_canny.png")


def figure3(image, pred, gt, fov_mask):
    """
    Figure 3: Original | Ground truth | TP/FN/FP error map (Canny).
    """
    vis = error_map(pred, gt, fov_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image);           axes[0].set_title("Original")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth")
    axes[2].imshow(vis);             axes[2].set_title("Prediction  (green=TP  red=FN  blue=FP)")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure3_mask_vs_gt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure3_mask_vs_gt.png")


LEGEND_ELEMENTS = [
    Patch(facecolor=(0/255,   200/255, 0/255),   label="TP — correct vessel"),
    Patch(facecolor=(220/255, 0/255,   0/255),   label="FN — missed vessel"),
    Patch(facecolor=(0/255,   0/255,   220/255), label="FP — false alarm"),
]


def figure4_2panel(image, pred_canny, pred_fusion, gt, fov_mask):
    """2-panel: Canny (worst) vs Fusion weighted (best) — compact for report."""
    maps = [(pred_canny, "Canny"), (pred_fusion, "Fusion (weighted)")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (pred, title) in zip(axes, maps):
        ax.imshow(error_map(pred, gt, fov_mask))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.legend(handles=LEGEND_ELEMENTS, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Error Maps — DRIVE Image 21 (Worst vs. Best Method)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure4_error_2panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure4_error_2panel.png")


def figure4_4panel(image, pred_canny, pred_gabor, pred_color, pred_fusion, gt, fov_mask):
    """2x2 grid: all four methods — for presentation slides."""
    maps = [
        (pred_canny,  "Canny"),
        (pred_gabor,  "Gabor"),
        (pred_color,  "Color Threshold"),
        (pred_fusion, "Fusion (weighted)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (pred, title) in zip(axes.flat, maps):
        ax.imshow(error_map(pred, gt, fov_mask))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.legend(handles=LEGEND_ELEMENTS, loc="lower center", ncol=3,
               fontsize=14, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Per-Method Error Maps — DRIVE Image 21", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure4_error_4panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure4_error_4panel.png")


def figure4_1row(image, pred_canny, pred_gabor, pred_color, pred_fusion, gt, fov_mask):
    """Single row: all four methods with minimal spacing."""
    maps = [
        (pred_canny,  "Canny"),
        (pred_gabor,  "Gabor"),
        (pred_color,  "Color Threshold"),
        (pred_fusion, "Fusion (weighted)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.subplots_adjust(wspace=0.08)
    for ax, (pred, title) in zip(axes, maps):
        ax.imshow(error_map(pred, gt, fov_mask))
        ax.set_title(title, fontsize=17)
        ax.axis("off")
    fig.legend(handles=LEGEND_ELEMENTS, loc="lower center", ncol=3,
               fontsize=17, frameon=True, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.5, handleheight=1.8, borderpad=1.0,
               labelspacing=1.0, handletextpad=0.8)
    fig.suptitle("Per-Method Error Maps — DRIVE Image 21", fontsize=18)
    fig.savefig(f"{OUT_DIR}/figure4_error_1row.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure4_error_1row.png")


if __name__ == "__main__":
    image, gt, fov_mask = load()

    edges, pred_canny  = run_canny(image)
    pred_gabor         = run_gabor(image)
    pred_color         = run_color(image)
    pred_fusion        = run_fusion_weighted(image)

    figure2(image, edges)
    figure3(image, pred_canny, gt, fov_mask)
    figure4_2panel(image, pred_canny, pred_fusion, gt, fov_mask)
    # figure4_4panel(image, pred_canny, pred_gabor, pred_color, pred_fusion, gt, fov_mask)
    figure4_1row(image, pred_canny, pred_gabor, pred_color, pred_fusion, gt, fov_mask)
