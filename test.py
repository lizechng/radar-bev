from PIL import Image
import numpy as np

im = Image.open('/media/personal_data/lizc/2023/radar-bev/mini1/val/lidHt/20211025_1_group0005_frame0000_1635145120_915.tiff')

im = np.asarray(im)
print(im.shape)
l = []
for i in range(600):
    for j in range(300):
        l.append(im[i,j])
print(im[0,0])
print(im[0,0].dtype)
# print(set(l))