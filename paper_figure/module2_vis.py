import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

before = np.load('../m2_before.npy')
before = np.load('../m2_after.npy')

# img = (before - np.min(before)) / (np.max(before) - np.min(before)) * 255
img = before / np.max(before) * 255 * 0.5
img = np.uint8(img)
img = img.T[::-1, ::]
print(np.min(img), np.max(img))
# Image.fromarray(img).show()
plt.imshow(img, cmap='jet')
# plt.colorbar()
plt.axis('off')
# Save the image
# plt.savefig('m2_after.png', dpi=300)
plt.show()