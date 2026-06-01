import numpy as np

def gaussian_kernel(size, sigma):
    """
    Build a 2D Gaussian kernel from scratch.

    Inputs:
        size: odd integer kernel size
        sigma: Gaussian standard deviation

    Output:
        kernel: (size, size) numpy array summing to 1
    """
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd")

    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = i - k
            y = j - k
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel /= np.sum(kernel)
    return kernel

def convolve2d(image, kernel):
    """
    Apply 2D convolution from scratch using zero padding.

    Inputs:
        image: 2D numpy array
        kernel: 2D numpy array

    Output:
        output: convolved image
    """
    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image, dtype=np.float32)

    flipped_kernel = np.flipud(np.fliplr(kernel))

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * flipped_kernel)

    return output

def gradient(image):
    """
    Compute image gradient using Sobel operators.

    Inputs:
        image: 2D grayscale image

    Outputs:
        magnitude: gradient magnitude
        direction: gradient direction in degrees, range [0, 180)
    """
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)   

    gx = convolve2d(image, sobel_x)
    gy = convolve2d(image, sobel_y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Convert radians to degrees and map to [0, 180)
    direction = np.rad2deg(np.arctan2(gy, gx))
    direction = (direction + 180) % 180

    return magnitude, direction

def non_maximum_suppression(magnitude, direction):
    """
    Thin edges by keeping only local maxima along gradient direction.

    Inputs:
        magnitude: gradient magnitude image
        direction: gradient direction in degrees

    Output:
        suppressed: thinned edge map
    """
    h, w = magnitude.shape
    suppressed = np.zeros((h, w), dtype=np.float32)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = direction[i, j]

            # Direction 0 degrees: compare left and right
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                before = magnitude[i, j - 1]
                after = magnitude[i, j + 1]

            # Direction 45 degrees: compare diagonal bottom-left and top-right
            elif 22.5 <= angle < 67.5:
                before = magnitude[i + 1, j - 1]
                after = magnitude[i - 1, j + 1]

            # Direction 90 degrees: compare top and bottom
            elif 67.5 <= angle < 112.5:
                before = magnitude[i - 1, j]
                after = magnitude[i + 1, j]

            # Direction 135 degrees: compare top-left and bottom-right
            else:
                before = magnitude[i - 1, j - 1]
                after = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= before and magnitude[i, j] >= after:
                suppressed[i, j] = magnitude[i, j]

    return suppressed


def hysteresis(edges, low, high):
    """
    Apply double thresholding and edge tracking by hysteresis.

    Inputs:
        edges: thinned edge magnitude image
        low: low threshold
        high: high threshold

    Output:
        result: binary edge map with values 0 or 1
    """
    h, w = edges.shape

    # If thresholds are given as ratios, scale by max edge value
    max_val = np.max(edges)
    if high <= 1.0:
        high = high * max_val
    if low <= 1.0:
        low = low * max_val

    strong = edges >= high
    weak = (edges >= low) & (edges < high)

    result = np.zeros((h, w), dtype=np.float32)
    result[strong] = 1.0

    visited = np.zeros((h, w), dtype=bool)

    # Start DFS/BFS from every strong edge
    strong_pixels = list(zip(*np.where(strong)))

    while strong_pixels:
        i, j = strong_pixels.pop()

        if visited[i, j]:
            continue

        visited[i, j] = True
        result[i, j] = 1.0

        # Check 8-connected neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni = i + di
                nj = j + dj

                if 0 <= ni < h and 0 <= nj < w:
                    if weak[ni, nj] and not visited[ni, nj]:
                        strong_pixels.append((ni, nj))

    return result


def canny(image, sigma=1.0, low=0.05, high=0.15, return_response=False):
    """
    Full Canny edge detector from scratch.

    Args:
        image: grayscale or RGB image
        sigma: Gaussian blur sigma
        low: low threshold ratio or absolute value
        high: high threshold ratio or absolute value
        return_response: if True, also return the normalized NMS magnitude
                         (useful as a continuous response map for ROC curves)

    Returns:
        edges: binary edge map
        response_map: normalized NMS magnitude in [0, 1] (only if return_response=True)
    """
    image = image.astype(np.float32)

    # Convert RGB to grayscale if needed
    if image.ndim == 3:
        image = (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        )

    # Normalize if image is in 0-255 range
    if np.max(image) > 1.0:
        image = image / 255.0

    # Common choice: kernel size about 6 sigma, rounded to odd
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1

    kernel = gaussian_kernel(size, sigma)
    smoothed = convolve2d(image, kernel)

    mag, direction = gradient(smoothed)
    thin = non_maximum_suppression(mag, direction)
    edges = hysteresis(thin, low, high)

    if return_response:
        max_val = thin.max()
        response_map = thin / (max_val + 1e-8)
        return edges, response_map

    return edges