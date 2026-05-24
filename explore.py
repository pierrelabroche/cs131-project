import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load one DRIVE image
image = np.array(Image.open('data/DRIVE/training/images/21_training.tif'))
gt    = np.array(Image.open('data/DRIVE/training/1st_manual/21_manual1.gif'))
mask  = np.array(Image.open('data/DRIVE/training/mask/21_training_mask.gif'))

print(image.shape)   # should be (584, 565, 3)
print(gt.shape)      # should be (584, 565)
print(mask.shape)    # should be (584, 565)
print(gt.max())      # should be 255

# Quick visualization
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[1].imshow(gt, cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(mask, cmap='gray')
axes[2].set_title('FOV Mask')
plt.savefig('results/figures/data_check.png')
plt.show()