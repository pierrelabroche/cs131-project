import numpy as np
import cv2


def color_threshold_segment(image_rgb, threshold=0.5):
    """
    Segment retinal vessels using color thresholding in Lab color space.

    Vessels appear darker than surrounding retinal tissue, which shows up
    clearly in the L (lightness) channel of Lab space. We invert L so vessels
    become bright, enhance with CLAHE, then apply a fixed threshold.

    Inputs:
        image_rgb: (H, W, 3) retinal image (any dtype — converted to uint8)
        threshold: float in [0, 1] applied to normalized enhanced L channel
    Outputs:
        binary mask (H, W) uint8, enhanced inverted L channel (H, W) uint8
    """
    # Ensure uint8 — skimage may load TIFFs as uint16
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb / image_rgb.max() * 255).astype(np.uint8)

    # Drop alpha channel if present
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]

    # Convert RGB to Lab color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Extract L channel (lightness, 0-255 in OpenCV encoding)
    L = lab[:, :, 0]

    # Invert so vessels (dark) become bright peaks
    L_inv = (255 - L).astype(np.uint8)

    # CLAHE to enhance vessel contrast before thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L_inv)

    # Normalize to [0, 1] and apply fixed threshold
    L_norm = L_enhanced / 255.0
    binary_mask = (L_norm >= threshold).astype(np.uint8)

    return binary_mask, L_enhanced
