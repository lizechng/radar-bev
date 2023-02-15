# Semantic-guided depth completion
import numpy as np
import cv2
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
import re
import warnings
import math
from scipy.linalg import pinv, inv

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    '''Parse header of PCD files'''
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn(f'warning: cannot understand line: {ln}')
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()

    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def _build_dtype(metadata):
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_ascii_pc_data(f, dtype, metadata):
    # for radar point
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    # for lidar point
    rowstep = metadata['points'] * dtype.itemsize
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    raise NotImplemented


def read_pcd(pcd_path, pts_view=False):
    # pcd = o3d.io.read_point_cloud(pcd_path)
    f = open(pcd_path, 'rb')
    header = []
    while True:
        ln = f.readline().strip()  # ln is bytes
        ln = str(ln, encoding='utf-8')
        header.append(ln)
        # print(type(ln), ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or "binary_compressed"')

    points = np.concatenate([pc_data[metadata['fields'][0]][:, None],
                             pc_data[metadata['fields'][1]][:, None],
                             pc_data[metadata['fields'][2]][:, None],
                             pc_data[metadata['fields'][3]][:, None]], axis=-1)
    # print(f'pcd points: {points.shape}')

    if pts_view:
        ptsview(points)
    return points


def ptsview(points):
    pass


def pts2camera(pts, calib_path, matrix=None):
    if matrix is None:
        try:
            matrix = json.load(open(calib_path))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
        except:
            matrix = json.load(open(calib_path))['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    matrix = np.asarray(matrix)

    n = pts.shape[0]
    pts = np.hstack((pts, np.ones((n, 1))))
    print(pts.shape, matrix.shape)
    pts_2d = np.dot(pts, np.transpose(matrix))
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    mask = (pts_2d[:, 0] < 3517) & (pts_2d[:, 1] < 1700) & \
           (pts_2d[:, 0] > 0) & (pts_2d[:, 1] > 0) & \
           (pts_2d[:, 2] > 0) & (pts_2d[:, 2] < 75)
    pts_2d = pts_2d[mask, :]
    print(pts_2d[:, 0].shape)
    x = np.asarray(pts_2d[:, 0] / 3517 * 512, dtype=np.int32)
    y = np.asarray(pts_2d[:, 1] / 1700 * 256, dtype=np.int32)
    v = np.asarray(pts_2d[:, 2], dtype=np.uint8)
    im = np.zeros([256, 512], dtype=np.uint8)
    im[y, x] = v

    return im


def pointcloud_transform(pointcloud, transform_matrix):
    '''
        transform pointcloud from coordinate1 to coordinate2 according to transform_matrix
    :param pointcloud: (x, y, z, ...)
    :param transform_matrix:
    :return pointcloud_transformed: (x, y, z, ...)
    '''
    n_points = pointcloud.shape[0]
    xyz = pointcloud[:, :3]
    xyz1 = np.vstack((xyz.T, np.ones((1, n_points))))
    xyz1_transformed = np.matmul(transform_matrix, xyz1)
    pointcloud_transformed = np.hstack((
        xyz1_transformed[:3, :].T,
        pointcloud[:, 3:]
    ))
    return pointcloud_transformed


def pts2rbev(lpts, calib_lid, calib_ti):
    # Remove LiDAR ground points
    from ransac import my_ransac_v3
    indices, model = my_ransac_v3(lpts[:, :3], distance_threshold=0.7)
    # ground_pts = lpts[indices]
    # ptsview(lpts)
    # ptsview(ground_pts)
    lpts = np.delete(lpts, indices, axis=0)
    # ptsview(lpts)

    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera1_TransformMatrix = json.load(open(calib_lid, 'r'))[
            'VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    TIRadar_to_LeopardCamera1_TransformMatrix = json.load(open(calib_ti, 'r'))[
            'TIRadar_to_LeopardCamera1_TransformMatrix']
    LeopardCamera1_IntrinsicMatrix = np.array(
        [
            [1976.27129878769, 0, 1798.25228491297],
            [0, 1977.80114435384, 1000.96808764067],
            [0, 0, 1]
        ]
    )
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )

    pts = pointcloud_transform(lpts, VelodyneLidar_to_TIRadar_TransformMatrix)

    # ptsview(lpts)
    # ptsview(pts)

    side_range = (-75, 75)
    fwd_range = (0, 75)
    height_range = (-30, 5)  # height_range should be modified manually
    y, x, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # print(np.min(x), np.max(x)) # -199.20, 197.77
    # print(np.min(y), np.max(y)) # -185.53, 196.74
    # print(np.min(z), np.max(z)) # -5.02, 39.75
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    h_filter = np.logical_and(z > height_range[0], z < height_range[1])
    filter = np.logical_and(f_filter, s_filter)
    filter = np.logical_and(filter, h_filter)  # height filter
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.25
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)

    # pixel_value = np.clip(a=z, a_max=height_range[1], a_min=height_range[0])
    pixel_value = 255  # z

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    # pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    # pixel_value = (pixel_value - np.min(pixel_value)) / (np.max(pixel_value) - np.min(pixel_value)) * 255
    x_max = int((side_range[1] - side_range[0]) / res) + 1
    y_max = int((fwd_range[1] - fwd_range[0]) / res) + 1

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    # im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    # print(im.shape)
    # plt.imshow(im.T[::-1,::-1])
    # plt.show()
    return im.T[::-1, ::-1][:600, :300]


def pts2rbev_fg(lpts, calib_lid, calib_ti, h_filter=1):

    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera1_TransformMatrix = json.load(open(calib_lid, 'r'))[
            'VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    TIRadar_to_LeopardCamera1_TransformMatrix = json.load(open(calib_ti, 'r'))[
            'TIRadar_to_LeopardCamera1_TransformMatrix']
    LeopardCamera1_IntrinsicMatrix = np.array(
        [
            [1976.27129878769, 0, 1798.25228491297],
            [0, 1977.80114435384, 1000.96808764067],
            [0, 0, 1]
        ]
    )
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )

    pts = pointcloud_transform(lpts, VelodyneLidar_to_TIRadar_TransformMatrix)

    # ptsview(lpts)
    # ptsview(pts)

    side_range = (-75, 75)
    fwd_range = (0, 75)
    height_range = (h_filter, 15)  # height_range should be modified manually
    y, x, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # print(np.min(x), np.max(x)) # -199.20, 197.77
    # print(np.min(y), np.max(y)) # -185.53, 196.74
    # print(np.min(z), np.max(z)) # -5.02, 39.75
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    h_filter = np.logical_and(z > height_range[0], z < height_range[1])
    filter = np.logical_and(f_filter, s_filter)
    filter = np.logical_and(filter, h_filter)  # height filter
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.25
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)

    # pixel_value = np.clip(a=z, a_max=height_range[1], a_min=height_range[0])
    pixel_value = 255  # z

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    # pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    # pixel_value = (pixel_value - np.min(pixel_value)) / (np.max(pixel_value) - np.min(pixel_value)) * 255
    x_max = int((side_range[1] - side_range[0]) / res) + 1
    y_max = int((fwd_range[1] - fwd_range[0]) / res) + 1

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    # im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    # print(im.shape)
    # plt.imshow(im.T[::-1,::-1])
    # plt.show()
    return im.T[::-1, ::-1][:600, :300]


def pts2rbevHeight(lpts, calib_lid, calib_ti):
    # Remove LiDAR ground points
    from ransac import my_ransac_v3
    indices, model = my_ransac_v3(lpts[:, :3], distance_threshold=0.7)
    # ground_pts = lpts[indices]
    # ptsview(lpts)
    # ptsview(ground_pts)
    lpts = np.delete(lpts, indices, axis=0)
    # ptsview(lpts)

    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera1_TransformMatrix = json.load(open(calib_lid, 'r'))[
            'VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    TIRadar_to_LeopardCamera1_TransformMatrix = json.load(open(calib_ti, 'r'))[
            'TIRadar_to_LeopardCamera1_TransformMatrix']
    LeopardCamera1_IntrinsicMatrix = np.array(
        [
            [1976.27129878769, 0, 1798.25228491297],
            [0, 1977.80114435384, 1000.96808764067],
            [0, 0, 1]
        ]
    )
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )

    pts = pointcloud_transform(lpts, VelodyneLidar_to_TIRadar_TransformMatrix)

    # ptsview(lpts)
    # ptsview(pts)

    side_range = (-75, 75)
    fwd_range = (0, 75)
    height_range = (-10, 30)  # height_range should be modified manually
    y, x, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # print(np.min(x), np.max(x)) # -199.20, 197.77
    # print(np.min(y), np.max(y)) # -185.53, 196.74
    # print(np.min(z), np.max(z)) # -5.02, 39.75
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    h_filter = np.logical_and(z > height_range[0], z < height_range[1])
    filter = np.logical_and(f_filter, s_filter)
    filter = np.logical_and(filter, h_filter)  # height filter
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.25
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)

    # pixel_value = np.clip(a=z, a_max=height_range[1], a_min=height_range[0])
    pixel_value = z + 10 # range: 0 ~ 40, in dataloader, h/40

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    # pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    # pixel_value = (pixel_value - np.min(pixel_value)) / (np.max(pixel_value) - np.min(pixel_value)) * 255
    x_max = int((side_range[1] - side_range[0]) / res) + 1
    y_max = int((fwd_range[1] - fwd_range[0]) / res) + 1

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    # im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    # print(im.shape)
    # plt.imshow(im.T[::-1,::-1])
    # plt.show()
    return im.T[::-1, ::-1][:600, :300]


def pts2bev(pts):
    side_range = (-75, 75)
    fwd_range = (0, 75)
    height_range = (-2, 5)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # print(np.min(x), np.max(x)) # -199.20, 197.77
    # print(np.min(y), np.max(y)) # -185.53, 196.74
    # print(np.min(z), np.max(z)) # -5.02, 39.75
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    h_filter = np.logical_and(z > height_range[0], z < height_range[1])
    filter = np.logical_and(f_filter, s_filter)
    filter = np.logical_and(filter, h_filter)  # height filter
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.25
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)

    # pixel_value = np.clip(a=z, a_max=height_range[1], a_min=height_range[0])
    pixel_value = 250  # z

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    # pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    # pixel_value = (pixel_value - np.min(pixel_value)) / (np.max(pixel_value) - np.min(pixel_value)) * 255
    x_max = int((side_range[1] - side_range[0]) / res) + 1
    y_max = int((fwd_range[1] - fwd_range[0]) / res) + 1

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    # im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    print(im.shape)
    # plt.imshow(im, cmap='jet')
    # plt.show()
    return im[:, :]


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


def radar_polar_to_cartesian(pth=None, cart_pixel_width=601, cart_pixel_height=301):
    # pth = './Dataset/20211027_1_group0021/group0021_frame0000/TIRadar/1635319097.410.heatmap.bin'
    res = loadTIRadarHeatmap(pth)
    # shape: 257, 232
    x_bins = res['x_bins']
    y_bins = res['y_bins']
    static = res['heatmap_static']
    dynamic = res['heatmap_dynamic']  # shape: 257, 232

    coords_x = np.linspace(-75, 75, cart_pixel_width, dtype=np.float32)
    coords_y = np.linspace(0, 75, cart_pixel_height, dtype=np.float32)
    Y, X = np.meshgrid(coords_y, coords_x)
    sample_range = np.sqrt(Y * Y + X * X)  # shape: 600, 300
    sample_angle = np.arctan2(Y, X) / np.pi * 180

    # Interpolate Radar Data Coordinates
    angle = (np.arctan2(y_bins, x_bins) / np.pi * 180)[:, 0]  # shape: 257,
    distance = np.sqrt(x_bins ** 2 + y_bins ** 2)[0, :]  # shape: 232,
    anglx = np.arange(0, 257)
    distancx = np.arange(0, 232)

    sample_u = np.interp(sample_range, distance, distancx).astype(np.float32)
    sample_v = np.interp(sample_angle, angle, anglx).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    ####### Heatmap normalization ######################
    hm = static + dynamic
    hm = np.uint8(hm / np.max(hm) * 255)
    hm = np.expand_dims(hm, -1)

    ####### Heatmap remap ##############################
    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(hm, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    cart_im = cart_img[:, :, 0]

    # return cart_im.T[::-1, ::]
    return cart_im[:600, :300]


def pltRadLid(radar, lidar):
    def norm_image(image):
        image = image.copy()
        image -= np.max(np.min(image), 0)
        image /= np.max(image)
        image *= 255
        return np.uint8(image)

    masks = norm_image(np.float32(lidar)).astype(np.uint8)
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_HOT)
    heatmap = np.float32(heatmap)

    cam = 0.2 * heatmap + 0.8 * np.float32(radar)
    Image.fromarray(np.uint8(cam)).show()
    return cam


# Generate Radar RF mask
def generate_mask(pth=None, cart_pixel_width=601, cart_pixel_height=301):
    # image's shape is 3517x1700
    # Camera InstrinsicMatrix
    # [1976.271299,           0, 1798.252285],
    # [          0, 1977.801144, 1000.968088],
    # [          0,           0,           1]
    lx, ly = 3517, 1700
    fx, fy = 1976.271299, 1977.801144
    cx, cy = 1798.252285, 1000.968088

    amax = math.atan2(lx - cx, fx) / math.pi * 180 + 90
    amin = math.atan2(0 - cx, fx) / math.pi * 180 + 90
    # print(amin, amax)
    # pth = './Dataset/20211027_1_group0021/group0021_frame0000/TIRadar/1635319097.410.heatmap.bin'
    res = loadTIRadarHeatmap(pth)
    # shape: 257, 232
    x_bins = res['x_bins']
    y_bins = res['y_bins']
    static = res['heatmap_static']
    dynamic = res['heatmap_dynamic']  # shape: 257, 232

    mask_value = np.zeros_like(static)

    coords_x = np.linspace(-75, 75, cart_pixel_width, dtype=np.float32)
    coords_y = np.linspace(0, 75, cart_pixel_height, dtype=np.float32)
    Y, X = np.meshgrid(coords_y, coords_x)
    sample_range = np.sqrt(Y * Y + X * X)  # shape: 600, 300
    sample_angle = np.arctan2(Y, X) / np.pi * 180

    # Interpolate Radar Data Coordinates
    angle = (np.arctan2(y_bins, x_bins) / np.pi * 180)[:, 0]  # shape: 257,
    distance = np.sqrt(x_bins ** 2 + y_bins ** 2)[0, :]  # shape: 232,
    anglx = np.arange(0, 257)
    distancx = np.arange(0, 232)

    # sample mask
    mask_angle = np.arctan2(y_bins, x_bins) / np.pi * 180
    for i in range(mask_angle.shape[0]):
        for j in range(mask_angle.shape[1]):
            if mask_angle[i, j] >= amin and mask_angle[i, j] <= amax:
                mask_value[i, j] = 255

    sample_u = np.interp(sample_range, distance, distancx).astype(np.float32)
    sample_v = np.interp(sample_angle, angle, anglx).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    ####### Heatmap normalization ######################
    hm = mask_value
    hm = np.uint8(hm / np.max(hm) * 255)
    hm = np.expand_dims(hm, -1)

    ####### Heatmap remap ##############################
    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(hm, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    cart_im = cart_img[:, :, 0]

    return cart_im[:600, :300]


def pts2cam_test(pts, img_path, calib_path, matrix=None):
    img = Image.open(img_path)
    if matrix is None:
        try:
            matrix = json.load(open(calib_path))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
        except:
            matrix = json.load(open(calib_path))['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    matrix = np.asarray(matrix)

    n = pts.shape[0]
    pts = np.hstack((pts, np.ones((n, 1))))
    print(pts.shape, matrix.shape)
    pts_2d = np.dot(pts, np.transpose(matrix))

    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    mask = (pts_2d[:, 0] < 3517) & (pts_2d[:, 1] < 1700) & \
           (pts_2d[:, 0] > 0) & (pts_2d[:, 1] > 0)
    pts_2d = pts_2d[mask, :]

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    image = np.asarray(img)
    for i in range(pts_2d.shape[0]):
        depth = pts_2d[i, 2]
        color = cmap[int(depth), :]
        cv2.circle(image, (int(np.round(pts_2d[i, 0])),
                           int(np.round(pts_2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)

    Image.fromarray(image).show()


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.
    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def object_mask(json_path, calib_ti):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros([600, 300], dtype=np.uint8)
    res = 75 / 300
    color = (255, 255, 255)  # RGB

    VelodyneLidar_to_LeopardCamera1_TransformMatrix = data['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    TIRadar_to_LeopardCamera1_TransformMatrix = json.load(open(calib_ti, 'r'))[
        'TIRadar_to_LeopardCamera1_TransformMatrix']
    LeopardCamera1_IntrinsicMatrix = np.array(
        [
            [1976.27129878769, 0, 1798.25228491297],
            [0, 1977.80114435384, 1000.96808764067],
            [0, 0, 1]
        ]
    )
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )
    P = VelodyneLidar_to_TIRadar_TransformMatrix
    # print(P.shape) # 4x4

    for ob in data['annotation']:
        R = rotz(ob['alpha'])

        # 3d bounding box dimensions
        l = ob['l']
        w = ob['w']
        h = ob['h']

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + ob['x']
        corners_3d[1, :] = corners_3d[1, :] + ob['y']
        corners_3d[2, :] = corners_3d[2, :] + ob['z']

        # LiDAR2Radar
        pointcloud = corners_3d.T
        n_points = pointcloud.shape[0]
        xyz = pointcloud[:, :3]
        xyz1 = np.vstack((xyz.T, np.ones((1, n_points))))
        xyz1_transformed = np.matmul(P, xyz1)
        pointcloud_transformed = np.hstack((
            xyz1_transformed[:3, :].T,
            pointcloud[:, 3:]
        ))

        pts = pointcloud_transformed[:4, [1, 0]] / res
        pts[:, 1] = pts[:, 1] + 300
        pts = pts.astype(np.int32)
        # pts = pts.reshape(4, 1, 2)
        cv2.fillPoly(mask, [pts], color=color)

    return mask


def save_data(base_path, num):
    groups = os.listdir(base_path)
    groups = sorted(groups)
    for group in groups[:num]:
        group_path = os.path.join(base_path, group)
        tmp = os.listdir(group_path)
        tmp = sorted(tmp)
        folders = [itm for itm in tmp if 'labeled' in itm]
        print(folders)
        # raise NotImplemented
        # folders = tmp

        sp = 'train'
        for folder in folders[0::2]:
            camera_path = os.path.join(group_path, folder, 'LeopardCamera1')
            for file in os.listdir(camera_path):
                if file[-3:] == 'png':
                    img_path = os.path.join(camera_path, file)
            lidar_path = os.path.join(group_path, folder, 'VelodyneLidar')
            for file in os.listdir(lidar_path):
                if file[-3:] == 'pcd':
                    pcd_lidar = os.path.join(lidar_path, file)
                if file[-4:] == 'json':
                    calib_lidar = os.path.join(lidar_path, file)
            radar_path = os.path.join(group_path, folder, 'OCULiiRadar')
            for file in os.listdir(radar_path):
                if file[-3:] == 'pcd':
                    pcd_radar = os.path.join(radar_path, file)
                if file[-4:] == 'json':
                    calib_radar = os.path.join(radar_path, file)

            ti_path = os.path.join(group_path, folder, 'TIRadar')
            for file in os.listdir(ti_path):
                if file[-3:] == 'pcd':
                    pcd_ti = os.path.join(ti_path, file)
                if file[-4:] == 'json':
                    calib_ti = os.path.join(ti_path, file)
                if file[-11:] == 'heatmap.bin':
                    hm_ti = os.path.join(ti_path, file)

            # LiDAR Depth
            depth_im = pts2camera(read_pcd(pcd_lidar)[:, :3], calib_lidar)
            # radar RF image, polar coordinate to cartesian coordiante
            cart_im = radar_polar_to_cartesian(hm_ti)
            # Bev image, where lidar points transformed into radar coordinate
            # lid_im = pts2rbev(read_pcd(pcd_lidar), calib_lidar, calib_ti)
            lid_origin = pts2rbev_fg(read_pcd(pcd_lidar), calib_lidar, calib_ti, h_filter=-10)
            lid_filter = pts2rbev_fg(read_pcd(pcd_lidar), calib_lidar, calib_ti, h_filter=1)
            lid_Ht = pts2rbevHeight(read_pcd(pcd_lidar), calib_lidar, calib_ti)
            # bev mask
            fov_mask = generate_mask(hm_ti)
            # lidar points in camera image
            # pts2cam_test(read_pcd(pcd_lidar)[:, :3], img_path, calib_lidar)
            # lidar in FOV
            # tmp = np.where(mask > 0, lid_im, 0)
            # Image.fromarray(np.uint8(tmp)).show()

            # cm = plt.get_cmap('jet')
            # colored_image = cm(cart_im)
            # pltRadLid(colored_image[:,:,:3]*255, lid_im)

            # object mask
            obj_mask = object_mask(calib_lidar, calib_ti)

            lid_im = lid_origin * (obj_mask / 255) + lid_filter * (1 - obj_mask / 255)

            # filename
            date = re.search(r'\d+_\d', img_path).group()
            frame = re.search(r'group\d\d\d\d_frame\d\d\d\d', img_path).group()
            timestamp = img_path.split('/')[-1][:-4].replace('.', '_')
            name = f'{date}_{frame}_{timestamp}'

            # Image
            img = Image.open(img_path).resize((1024, 512), resample=0)
            img = np.asarray(img)
            Image.fromarray(np.uint8(img)).save(f'./dataset/{sp}/img/{name}.png')
            # Radar
            Image.fromarray(np.uint8(cart_im)).save(f'./dataset/{sp}/radar/{name}.png')
            # LiDAR
            Image.fromarray(np.uint8(lid_im)).save(f'./dataset/{sp}/lidar/{name}.png')
            Image.fromarray(np.uint8(lid_Ht)).save(f'./dataset/{sp}/lidHt/{name}.png')
            # fov mask
            Image.fromarray(np.uint8(fov_mask)).save(f'./dataset/{sp}/fov_mask/{name}.png')
            # obj mask
            Image.fromarray(np.uint8(obj_mask)).save(f'./dataset/{sp}/obj_mask/{name}.png')
            # depth
            Image.fromarray(np.uint8(depth_im)).save(f'./dataset/{sp}/depth/{name}.png')
            # calib TI
            matrix = np.asarray(json.load(open(calib_ti, 'r'))['TIRadar_to_LeopardCamera1_TransformMatrix'])
            m = np.vstack([matrix, np.asarray([0, 0, 0, 1])])
            calib = {'Cam2TI': inv(m).tolist()}
            with open(f'./dataset/{sp}/calib/{name}.json', 'w') as f:
                json.dump(calib, f)

        sp = 'val'
        for folder in folders[1::2]:
            camera_path = os.path.join(group_path, folder, 'LeopardCamera1')
            for file in os.listdir(camera_path):
                if file[-3:] == 'png':
                    img_path = os.path.join(camera_path, file)
            lidar_path = os.path.join(group_path, folder, 'VelodyneLidar')
            for file in os.listdir(lidar_path):
                if file[-3:] == 'pcd':
                    pcd_lidar = os.path.join(lidar_path, file)
                if file[-4:] == 'json':
                    calib_lidar = os.path.join(lidar_path, file)
            radar_path = os.path.join(group_path, folder, 'OCULiiRadar')
            for file in os.listdir(radar_path):
                if file[-3:] == 'pcd':
                    pcd_radar = os.path.join(radar_path, file)
                if file[-4:] == 'json':
                    calib_radar = os.path.join(radar_path, file)

            ti_path = os.path.join(group_path, folder, 'TIRadar')
            for file in os.listdir(ti_path):
                if file[-3:] == 'pcd':
                    pcd_ti = os.path.join(ti_path, file)
                if file[-4:] == 'json':
                    calib_ti = os.path.join(ti_path, file)
                if file[-11:] == 'heatmap.bin':
                    hm_ti = os.path.join(ti_path, file)

            # LiDAR Depth
            depth_im = pts2camera(read_pcd(pcd_lidar)[:, :3], calib_lidar)
            # radar RF image, polar coordinate to cartesian coordiante
            cart_im = radar_polar_to_cartesian(hm_ti)
            # Bev image, where lidar points transformed into radar coordinate
            # lid_im = pts2rbev(read_pcd(pcd_lidar), calib_lidar, calib_ti)
            lid_origin = pts2rbev_fg(read_pcd(pcd_lidar), calib_lidar, calib_ti, h_filter=-10)
            lid_filter = pts2rbev_fg(read_pcd(pcd_lidar), calib_lidar, calib_ti, h_filter=1)
            lid_Ht = pts2rbevHeight(read_pcd(pcd_lidar), calib_lidar, calib_ti)
            # bev mask
            fov_mask = generate_mask(hm_ti)
            # lidar points in camera image
            # pts2cam_test(read_pcd(pcd_lidar)[:, :3], img_path, calib_lidar)
            # lidar in FOV
            # tmp = np.where(mask > 0, lid_im, 0)
            # Image.fromarray(np.uint8(tmp)).show()

            # cm = plt.get_cmap('jet')
            # colored_image = cm(cart_im)
            # pltRadLid(colored_image[:,:,:3]*255, lid_im)

            # object mask
            obj_mask = object_mask(calib_lidar, calib_ti)

            lid_im = lid_origin * (obj_mask / 255) + lid_filter * (1 - obj_mask / 255)
            # filename
            date = re.search(r'\d+_\d', img_path).group()
            frame = re.search(r'group\d\d\d\d_frame\d\d\d\d', img_path).group()
            timestamp = img_path.split('/')[-1][:-4].replace('.', '_')
            name = f'{date}_{frame}_{timestamp}'
            # Image
            img = Image.open(img_path).resize((1024, 512), resample=0)
            img = np.asarray(img)
            Image.fromarray(np.uint8(img)).save(f'./dataset/{sp}/img/{name}.png')
            # Radar
            Image.fromarray(np.uint8(cart_im)).save(f'./dataset/{sp}/radar/{name}.png')
            # LiDAR
            Image.fromarray(np.uint8(lid_im)).save(f'./dataset/{sp}/lidar/{name}.png')
            Image.fromarray(np.uint8(lid_Ht)).save(f'./dataset/{sp}/lidHt/{name}.png')
            # fov mask
            Image.fromarray(np.uint8(fov_mask)).save(f'./dataset/{sp}/fov_mask/{name}.png')
            # obj mask
            Image.fromarray(np.uint8(obj_mask)).save(f'./dataset/{sp}/obj_mask/{name}.png')
            fov_mask = generate_mask(hm_ti)
            # lidar points in camera image
            # pts2cam_test(read_pcd(pcd_lidar)[:, :3], img_path, calib_lidar)
            # lidar in FOV
            # tmp = np.where(mask > 0, lid_im, 0)
            # Image.fromarray(np.uint8(tmp)).show()

            # cm = plt.get_cmap('jet')
            # colored_image = cm(cart_im)
            # pltRadLid(colored_image[:,:,:3]*255, lid_im)

            # object mask
            obj_mask = object_mask(calib_lidar, calib_ti)

            # filename
            date = re.search(r'\d+_\d', img_path).group()
            frame = re.search(r'group\d\d\d\d_frame\d\d\d\d', img_path).group()
            timestamp = img_path.split('/')[-1][:-4].replace('.', '_')
            name = f'{date}_{frame}_{timestamp}'
            # Image
            img = Image.open(img_path).resize((1024, 512), resample=0)
            img = np.asarray(img)
            Image.fromarray(np.uint8(img)).save(f'./dataset/{sp}/img/{name}.png')
            # Radar
            Image.fromarray(np.uint8(cart_im)).save(f'./dataset/{sp}/radar/{name}.png')
            # LiDAR
            Image.fromarray(np.uint8(lid_im)).save(f'./dataset/{sp}/lidar/{name}.png')
            # fov mask
            Image.fromarray(np.uint8(fov_mask)).save(f'./dataset/{sp}/fov_mask/{name}.png')
            # obj mask
            Image.fromarray(np.uint8(obj_mask)).save(f'./dataset/{sp}/obj_mask/{name}.png')
            # depth
            Image.fromarray(np.uint8(depth_im)).save(f'./dataset/{sp}/depth/{name}.png')
            # calib TI
            matrix = np.asarray(json.load(open(calib_ti, 'r'))['TIRadar_to_LeopardCamera1_TransformMatrix'])
            m = np.vstack([matrix, np.asarray([0, 0, 0, 1])])
            calib = {'Cam2TI': inv(m).tolist()}
            with open(f'./dataset/{sp}/calib/{name}.json', 'w') as f:
                json.dump(calib, f)

if __name__ == '__main__':
    # origin data folder structure
    # |--base_dir
    # |----20211025_1_group0005
    # |------20211025_1_group0005_frame0001
    # |--------LeopardCamera1
    # |--------MEMS
    # |--------OCULiiRadar
    # |--------TIRadar
    # |--------VelodyneLidar
    # training dataset structure
    # |--dataset
    # |----train
    # |------img
    # |------radar
    # |------lidar
    # create 'train' dirs
    if not os.path.exists('./dataset/train/img'):
        os.makedirs('./dataset/train/img', exist_ok=True)
    if not os.path.exists('./dataset/train/depth'):
        os.makedirs('./dataset/train/depth', exist_ok=True)
    if not os.path.exists('./dataset/train/radar'):
        os.makedirs('./dataset/train/radar', exist_ok=True)
    if not os.path.exists('./dataset/train/lidar'):
        os.makedirs('./dataset/train/lidar', exist_ok=True)
    if not os.path.exists('./dataset/train/lidHt'):
        os.makedirs('./dataset/train/lidHt', exist_ok=True)
    if not os.path.exists('./dataset/train/fov_mask'):
        os.makedirs('./dataset/train/fov_mask', exist_ok=True)
    if not os.path.exists('./dataset/train/obj_mask'):
        os.makedirs('./dataset/train/obj_mask', exist_ok=True)
    if not os.path.exists('./dataset/train/calib'):
        os.makedirs('./dataset/train/calib', exist_ok=True)
    # create 'val' dirs
    if not os.path.exists('./dataset/val/img'):
        os.makedirs('./dataset/val/img', exist_ok=True)
    if not os.path.exists('./dataset/val/depth'):
        os.makedirs('./dataset/val/depth', exist_ok=True)
    if not os.path.exists('./dataset/val/radar'):
        os.makedirs('./dataset/val/radar', exist_ok=True)
    if not os.path.exists('./dataset/val/lidar'):
        os.makedirs('./dataset/val/lidar', exist_ok=True)
    if not os.path.exists('./dataset/val/lidHt'):
        os.makedirs('./dataset/val/lidHt', exist_ok=True)
    if not os.path.exists('./dataset/val/fov_mask'):
        os.makedirs('./dataset/val/fov_mask', exist_ok=True)
    if not os.path.exists('./dataset/val/obj_mask'):
        os.makedirs('./dataset/val/obj_mask', exist_ok=True)
    if not os.path.exists('./dataset/val/calib'):
        os.makedirs('./dataset/val/calib', exist_ok=True)

    base_path = '/media/ourDataset/v1.0_label/'
    save_data(base_path, 3)

