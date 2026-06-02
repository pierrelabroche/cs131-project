import numpy as np
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes
from canny import canny
from gabor import gabor_segment
from color_threshold import color_threshold_segment


def _get_response_maps(img, enhanced):
    """Collect normalized response maps from all three base methods."""
    edges = canny(enhanced, sigma=1.0, low=0.05, high=0.15)
    canny_response = binary_fill_holes(closing(edges, disk(2))).astype(np.float32)
    _, gabor_response = gabor_segment(enhanced)
    _, L_enhanced = color_threshold_segment(img)
    color_response = L_enhanced.astype(np.float32) / 255.0
    return canny_response, gabor_response, color_response


def average_fusion_segment(img, enhanced, threshold=0.253):
    """
    Fuse Canny, Gabor, and color threshold by averaging their response maps.

    Inputs:
        img:      (H, W, 3) original RGB image
        enhanced: (H, W) CLAHE-enhanced green channel
        threshold: float in [0, 1] applied to the averaged map
    Outputs:
        binary_mask: (H, W) uint8
        fused_map:   (H, W) float in [0, 1]
    """
    response_maps = _get_response_maps(img, enhanced)
    fused_map = np.mean(np.stack(response_maps, axis=0), axis=0)
    binary_mask = (fused_map >= threshold).astype(np.uint8)
    return binary_mask, fused_map


def max_fusion_segment(img, enhanced, threshold=0.5):
    """
    Fuse by taking the element-wise max across response maps (logical OR).
    A pixel is vessel if *any* method gives a strong response — maximizes recall.

    Inputs:
        img:      (H, W, 3) original RGB image
        enhanced: (H, W) CLAHE-enhanced green channel
        threshold: float in [0, 1] applied to the max map
    Outputs:
        binary_mask: (H, W) uint8
        fused_map:   (H, W) float in [0, 1]
    """
    response_maps = _get_response_maps(img, enhanced)
    fused_map = np.max(np.stack(response_maps, axis=0), axis=0)
    binary_mask = (fused_map >= threshold).astype(np.uint8)
    return binary_mask, fused_map


def min_fusion_segment(img, enhanced, threshold=0.5):
    """
    Fuse by taking the element-wise min across response maps (logical AND).
    A pixel is vessel only if *all* methods agree — maximizes precision.

    Inputs:
        img:      (H, W, 3) original RGB image
        enhanced: (H, W) CLAHE-enhanced green channel
        threshold: float in [0, 1] applied to the min map
    Outputs:
        binary_mask: (H, W) uint8
        fused_map:   (H, W) float in [0, 1]
    """
    response_maps = _get_response_maps(img, enhanced)
    fused_map = np.min(np.stack(response_maps, axis=0), axis=0)
    binary_mask = (fused_map >= threshold).astype(np.uint8)
    return binary_mask, fused_map


# AUC-derived weights from per-image mean AUC on DRIVE training set:
# Canny=0.5002, Gabor=0.8023, Color=0.7277  (sum=2.0302)
AUC_WEIGHTS = np.array([0.5002, 0.8023, 0.7277])
AUC_WEIGHTS = AUC_WEIGHTS / AUC_WEIGHTS.sum()


def weighted_fusion_segment(img, enhanced, threshold=0.273):
    """
    Fuse Canny, Gabor, and Color Threshold weighted by their per-image mean AUC.
    Higher-AUC methods contribute more to the fused response map.

    Inputs:
        img:      (H, W, 3) original RGB image
        enhanced: (H, W) CLAHE-enhanced green channel
        threshold: float in [0, 1] applied to the weighted map
    Outputs:
        binary_mask: (H, W) uint8
        fused_map:   (H, W) float in [0, 1]
    """
    response_maps = _get_response_maps(img, enhanced)
    fused_map = sum(w * r for w, r in zip(AUC_WEIGHTS, response_maps))
    binary_mask = (fused_map >= threshold).astype(np.uint8)
    return binary_mask, fused_map
