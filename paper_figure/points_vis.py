import matplotlib.pyplot as plt
import numpy as np

points = np.load('../points.npy')
print(points.shape)

x, y, z = points[:, 0], points[:, 1], points[:, 2]

fig = plt.figure(dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o', s=2, linewidth=0, alpha=1, cmap='spectral')
# ax.axis('scaled')
plt.show()