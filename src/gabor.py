import numpy as np
import cv2

def gabor_segment(image_enhanced, threshold=0.1):
    '''
    Apply Gabor filter bank to the enhanced image (simplified Soares et al. 2006).
    Uses paper's scales (a=2,3,4,6) and 18 orientations (0-170 deg, step 10).
    Replaces their GMM classifier with direct thresholding of the max response map.

    Inputs:
        image_enhanced: 2D numpy array (CLAHE-enhanced green channel)
        threshold: float in [0, 1], applied to normalized response map (default 0.1).
                   Otsu tends to over-threshold because vessels are ~12% of pixels.
    Outputs:
        binary mask (H, W) uint8, response_map (H, W) float in [0, 1]
    '''
    # Soares et al. 2006 parameters
    orientations = np.arange(0, 180, 10)   # 18 angles
    wavelengths  = [2, 3, 4, 6]            # scales in pixels (paper: a=2,3,4,6)

    img = image_enhanced.astype(np.float64)

    responses = []
    # Compute Gabor responses for each (theta, lambda) pair
    # g(x, y) = exp(-(x'^2 + gamma^2 * y'^2) / (2 * sigma^2)) * cos(2 * pi * x' / lambda + psi)
    for lam in wavelengths:
      sigma = lam * 0.56
      gamma = 0.5
      size  = int(6 * sigma) | 1 # Ensure size is odd
      half  = size // 2

      x_coords, y_coords = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
      scale_responses = np.zeros_like(img)
      for theta in orientations:
          theta_rad = np.deg2rad(theta)
          x_prime = x_coords * np.cos(theta_rad) + y_coords * np.sin(theta_rad)
          y_prime = -x_coords * np.sin(theta_rad) + y_coords * np.cos(theta_rad)

          gabor_kernel = np.exp(-(x_prime**2 + (gamma**2) * y_prime**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_prime / lam)
          response = np.abs(cv2.filter2D(img, -1, gabor_kernel))  # Use absolute value of response

          scale_responses = np.maximum(scale_responses, response)  # Max response across orientations
      responses.append(scale_responses)

    # Combine scales: max across all scale response maps                                                                                                                                            
    response_map = np.max(np.stack(responses), axis=0) 

    # Normalize response map to [0, 1]
    response_map = (response_map - response_map.min()) / (response_map.max() - response_map.min() + 1e-8)
    binary_mask = (response_map >= threshold).astype(np.uint8)

    return binary_mask, response_map