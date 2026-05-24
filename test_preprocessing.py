import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from preprocessing import preprocess

# Load image
image = np.array(Image.open('data/DRIVE/training/images/21_training.tif'))
print(image.shape)  # image shape

# Run preprocessing
green, enhanced = preprocess(image)

# Save figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(green, cmap='gray')
axes[1].set_title('Green Channel')
axes[1].axis('off')

axes[2].imshow(enhanced, cmap='gray')
axes[2].set_title('CLAHE Enhanced')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('results/figures/preprocessing.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to results/figures/preprocessing.png")