import numpy as np
import cv2

def preprocess(image_rgb):
    """
    Input:  (H, W, 3) or (H, W, 4) RGB/RGBA retinal image
    Output: green channel, CLAHE enhanced green channel
    """
    # Handle RGBA — drop alpha channel if present
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]

    # Step 1: extract green channel
    green = image_rgb[:, :, 1]

    # Step 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    return green, enhanced