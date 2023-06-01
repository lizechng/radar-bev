import numpy as np

ti_pts = np.load('landmark_pts/landmark_ti.npy')
pred_pts = np.load('landmark_pts/landmark_pred_knn.npy')  # clutter points: 1m 109/2062

valid_inds = (pred_pts[:, 0] > -75) * \
             (pred_pts[:, 0] < 75) * \
             (pred_pts[:, 1] > 5) * \
             (pred_pts[:, 1] < 75) * \
             (pred_pts[:, 2] > -5) * \
             (pred_pts[:, 2] < 10)
pred_pts = pred_pts[valid_inds]

pts = []
for p in pred_pts:
    x, y, z = p[0], p[1], p[2]
    d = np.sqrt((ti_pts[:, 0] - x) ** 2 + (ti_pts[:, 1] - y) ** 2 + (ti_pts[:, 2] - z) ** 2)
    if min(d) < 1.0:
        continue
    pts.append(p)

pts = np.asarray(pts)
print(pred_pts.shape, pts.shape, ti_pts.shape)

np.save('landmark_pts/KNN_before.npy', pts)
