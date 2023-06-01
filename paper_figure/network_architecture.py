import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('../dataset_direct/train/radar/20211025_1_group0005_frame0030_1635145123_820.png')
img = np.asarray(img)
print(img.shape)

img = img.T[::-1, ::]
plt.imshow(img, cmap='jet')
# plt.colorbar()
plt.axis('off')
# Save the image
plt.savefig('radar_bev.png', dpi=300)
plt.show()

