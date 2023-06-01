import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('../dataset_direct/train/hessian/20211025_1_group0005_frame0020_1635145122_851.png')
img = np.asarray(img)
img = img.T[::-1, ::]
plt.imshow(img, cmap='jet')
# plt.colorbar()
plt.axis('off')
# Save the image
plt.savefig('after_module_1.png', dpi=300)
plt.show()

img = Image.open('../dataset_direct/train/radar/20211025_1_group0005_frame0020_1635145122_851.png')
img = np.asarray(img)
img = img.T[::-1, ::]
plt.imshow(img, cmap='jet')
# plt.colorbar()
plt.axis('off')
# Save the image
plt.savefig('before_module_1.png', dpi=300)
plt.show()

