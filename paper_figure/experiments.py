import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mayavi.mlab as mlab

metrix_ti = np.asarray(
    [[0.0005058522126699925, 4.0004836907199526e-07, -0.885619484742134, 0.2067938508885497],
     [-1.231040888565826e-05, -3.266686113720385e-05, 1.05244978435373, -0.002370372637584245],
     [1.1980997508422571e-06, -0.0005045554623847184, 0.4383194290194214, -0.05773464810825956],
     [0.0, 0.0, 0.0, 1.0]]
)

img = Image.open('/media/personal_data/lizc/2023/landmark_data/sparse-depth-completion-440.png')
gt = Image.open('/media/personal_data/lizc/2023/landmark_data/sparse-depth-completion-target.png')
img = np.asarray(img)
gt = np.asarray(gt)
plt.imshow(gt, cmap='jet')
plt.show()
print(np.min(img), np.max(img))
pts = []
for u in range(250, 400):  # 512
    for v in range(img.shape[1]):  # 1024
        if gt[u, v] == 0:
            continue
        if img[u, v] >= 75:
            continue
        if u >= 300:
            # if gt[u, v] != 0:
            d = (gt[u, v] * 0.8 + img[u, v] * 0.2)
            # d = img[u, v]
            pts.append([v / 1024 * 3517 * d, u / 512 * 1700 * d, d, 1])
        else:
            pts.append([v / 1024 * 3517 * img[u, v], u / 512 * 1700 * img[u, v], img[u, v], 1])

pts_2d = np.asarray(pts)
print(pts_2d.shape)

pts_3d = np.dot(pts_2d, np.transpose(metrix_ti))

print(np.min(pts_3d[:, 0]), np.max(pts_3d[:, 0]))
print(np.min(pts_3d[:, 1]), np.max(pts_3d[:, 1]))
print(np.min(pts_3d[:, 2]), np.max(pts_3d[:, 2]))

# Filter points
valid_inds = (pts_3d[:, 0] > -75) * \
             (pts_3d[:, 0] < 75) * \
             (pts_3d[:, 1] > 5) * \
             (pts_3d[:, 1] < 75) * \
             (pts_3d[:, 2] > -5) * \
             (pts_3d[:, 2] < 10)
pts_3d = pts_3d[valid_inds]
print(pts_3d.shape)
np.save('landmark_pts/gt_8_2-440.npy', pts_3d)
mlab.points3d(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
              np.sqrt(pts_3d[:, 0] ** 2 + pts_3d[:, 1] ** 2),
              mode='point',
              colormap='gnuplot', scale_factor=1)

# draw origin
mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

# draw axis
axes = np.array([
    [2., 0., 0., 0.],
    [0., 2., 0., 0.],
    [0., 0., 2., 0.],
], dtype=np.float64)
mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None,
            )
mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None,
            )
mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None,
            )

# draw fov (todo: update to real sensor spec.)
fov = np.array([  # 45 degree
    [20., 20., 0., 0.],
    [-20., 20., 0., 0.],
], dtype=np.float64)

mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
            )
mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
            )

# draw square region
TOP_Y_MIN = 0
TOP_Y_MAX = 75
TOP_X_MIN = -75
TOP_X_MAX = 75
TOP_Z_MIN = -2.0
TOP_Z_MAX = 0.4

x1 = TOP_X_MIN
x2 = TOP_X_MAX
y1 = TOP_Y_MIN
y2 = TOP_Y_MAX
mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )
mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )
mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )
mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )

mlab.show()

