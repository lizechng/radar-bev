import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist

lidar_pts = np.load('landmark_pts/landmark_lidar.npy') # Whether to filter ground points
ti_pts = np.load('landmark_pts/landmark_ti.npy')
pred_pts = np.load('landmark_pts/landmark_pred_knn.npy')  # clutter points: 1m 109/2062
# pred_pts = np.load('landmark_pts/landmark_pred_h.npy') # clutter points: 1m 0/1403
sparse_pts = np.load('landmark_pts/sparse_completion_8_2.npy')[:, :3]
sparse2_pts = np.load('landmark_pts/sparse_completion_8_1.npy')[:, :3]
for ep in [1]:
    LSS = np.load(f'landmark_pts/landmark_pred_knn.npy')

    print(f'lidar :{lidar_pts.shape}\n'
          f'ti    :{ti_pts.shape}\n'
          f'pred  :{pred_pts.shape}\n'
          f'sparse:{sparse_pts.shape}\n')

    target_pts = LSS

    valid_inds = (target_pts[:, 0] > -75) * \
                 (target_pts[:, 0] < 75) * \
                 (target_pts[:, 1] > 5) * \
                 (target_pts[:, 1] < 75) * \
                 (target_pts[:, 2] > -5) * \
                 (target_pts[:, 2] < 10)
    target_pts = target_pts[valid_inds]

    N = target_pts.shape[0]

    # print(f'target points: {N}')

    s_recall, s_clutter, D_H, D_C = 0, 0, 0, 0

    threshold = 1
    cnt = 0
    for p in target_pts:
        x, y, z = p[0], p[1], p[2]
        d = np.sqrt((lidar_pts[:, 0] - x) ** 2 + (lidar_pts[:, 1] - y) ** 2 + (lidar_pts[:, 2] - z) ** 2)
        if min(d) > threshold:
            cnt = cnt + 1
    s_clutter = cnt / N
    # print(f'radar clutter points: {cnt}, {s_clutter:.3f}\n')

    threshold = 1.5
    cnt = 0
    for p in lidar_pts:
        x, y, z = p[0], p[1], p[2]
        d = np.sqrt((target_pts[:, 0] - x) ** 2 + (target_pts[:, 1] - y) ** 2 + (target_pts[:, 2] - z) ** 2)
        if min(d) <= threshold:
            cnt = cnt + 1
    s_recall = cnt / N
    # print(f'radar recall points: {cnt}, {s_recall:.3f}\n')

    D_H = directed_hausdorff(target_pts, lidar_pts)[0]
    # print(f"Hausdorff distance: {D_H:.3f}")

    # Calculate Chamfer distance
    dist_a_to_b = cdist(target_pts, lidar_pts)
    dist_b_to_a = cdist(lidar_pts, target_pts)
    D_C = (dist_a_to_b.min(axis=1).mean() + dist_b_to_a.min(axis=1).mean()) / 2
    # print(f"Chamfer distance: {D_C:.3f}")

    print(f'{ep:03d}: {s_recall:.3f}, {s_clutter:.3f}, {D_H:.3f}, {D_C:.3f}, {N}')
