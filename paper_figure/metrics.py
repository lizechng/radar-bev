import numpy as np
import os


# Select the landmark frame.
# 20211025_1_group0005_frame0100_1635145130_809.npy *
# 20211025_1_group0009_frame0030_1635145184_906.npy * (For module 3)
# 20211025_1_group0009_frame0040_1635145185_936.npy
# 20211025_1_group0009_frame0050_1635145186_928.npy
# 20211025_1_group0009_frame0060_1635145187_930.npy
# 20211025_1_group0009_frame0070_1635145188_930.npy * ()
# 20211025_1_group0012_frame0010_1635145265_961.npy
# 20211025_1_group0012_frame0040_1635145268_982.npy
# 20211025_2_group0001_frame0100_1635151447_961.npy * (For module 3)
# 20211025_2_group0001_frame0110_1635151448_987.npy
# 20211025_2_group0001_frame0120_1635151449_940.npy
# 20211025_2_group0007_frame0490_1635151739_124.npy
# ...
# files = os.listdir('pts/radar')
# files = sorted(files)
# ti_number = []
# for file in files:
#     ti_pts = np.load(os.path.join('pts/radar', file))
#     lidar_pts = np.load(os.path.join('pts/lidar', file))
#     radar_pts = np.load(os.path.join('pts/pred', file))
#     if ti_pts.shape[0] > 200 and radar_pts.shape[0] > 800:
#         print(file)
#         print(lidar_pts.shape, radar_pts.shape, ti_pts.shape)
#         # np.save
#     ti_number.append(ti_pts.shape[0])

# Save the landmark frame
name = '20211025_2_group0001_frame0100_1635151447_961.npy'
lidar_pts = np.load(os.path.join('pts/lidar', name))
pred_pts = np.load(os.path.join('pts/pred', name))
pred_h_pts = np.load(os.path.join('pts/pred_h', name))
pred_knn_pts = np.load(os.path.join('pts/pred_knn', name))
pred_h_knn_pts = np.load(os.path.join('pts/pred_h_knn', name))
ti_pts = np.load(os.path.join('pts/radar', name))

print(np.min(lidar_pts[:, 0]), np.max(lidar_pts[:, 0]))
print(np.min(lidar_pts[:, 1]), np.max(lidar_pts[:, 1]))
print(np.min(lidar_pts[:, 2]), np.max(lidar_pts[:, 2]))

np.save(f'landmark_pts/landmark_lidar.npy', lidar_pts)
np.save(f'landmark_pts/landmark_pred.npy', pred_pts)
np.save(f'landmark_pts/landmark_pred_knn.npy', pred_knn_pts)
np.save(f'landmark_pts/landmark_pred_h.npy', pred_h_pts)
np.save(f'landmark_pts/landmark_pred_h_knn.npy', pred_h_knn_pts)
np.save(f'landmark_pts/landmark_ti.npy', ti_pts)

# The number of KNN optimized points is equal to the square of depth map.
# Thus, it should be optimized one more.
# But the recall points, and the precision points may just approciate.
lidar_pts = np.load('lidar.npy')
radar_pts = np.load('../points.npy')
# radar_pts = np.load('../with_knn.npy')
ti_pts = np.load('ti.npy')

print(f'lidar pts    : {lidar_pts.shape}\n'
      f'generated pts: {radar_pts.shape}\n'
      f'ti pts       : {ti_pts.shape}')

# Filter the radar points (with KNN)
# Y is the forward axis
# That is, the points after knn will less than generated points.
valid_inds = (radar_pts[:, 0] > -75) * \
             (radar_pts[:, 0] < 75) * \
             (radar_pts[:, 1] > 0) * \
             (radar_pts[:, 1] < 75) * \
             (radar_pts[:, 2] > -10) * \
             (radar_pts[:, 2] < 30)
radar_pts = radar_pts[valid_inds]
print(f'knn pts: {radar_pts.shape}')

# What about the points number of BASELINE.

# Metrics for the whole point cloud

# Metrics for the car point cloud
