import numpy as np
import matplotlib.pyplot as plt

def loadTIRadarHeatmap(heatmap_path):
    '''
    read TI radar heatmap
    :param heatmap_path: str - TI radar heatmap path
    :return: dict(np.array)
    '''
    data = np.fromfile(heatmap_path, dtype='float32')
    # print(data.shape)
    data = data.reshape((4 * 257, 232), order='F')
    data = data.reshape((4, 257, 232))
    res = {
        "heatmap_static": data[0, :, :],
        "heatmap_dynamic": data[1, :, :],
        "x_bins": data[2, :, :],
        "y_bins": data[3, :, :],
    }
    return res

res = loadTIRadarHeatmap('/media/personal_data/lizc/2/p2/labeled_data/20211025_1_group0005_155frames_31labeled/20211025_1_group0005_frame0030_labeled/TIRadar/1635145123.321.heatmap.bin')
# shape: 257, 232
x_bins = res['x_bins']
y_bins = res['y_bins']
static = res['heatmap_static']
dynamic = res['heatmap_dynamic']

img = dynamic.T[::-1, ::]
plt.imshow(img, cmap='jet')
# plt.colorbar()
plt.axis('off')
# Save the image
plt.savefig('radar_RAM.png', dpi=300)
plt.show()

