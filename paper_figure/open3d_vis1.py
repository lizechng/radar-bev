import open3d as o3d
import numpy as np
from PIL import Image

def main():
    raw_point = np.load('points.npy')  # 读取1.npy数据  N*[x,y,z]

    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="kitti")
    # 设置点云大小
    vis.get_render_option().point_size = 1.7
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([255/255, 255/255, 255/255])

    # ctr = vis.get_view_control()
    # ctr.set_lookat(np.array([0.0, 0.0, 55.0]))
    # ctr.set_up((0, -1, 0))  # set the positive direction of the x-axis as the up direction
    # ctr.set_front((-1, 0, 0))  # set the positive direction of the x-axis toward you


    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)

    # 设置点的颜色为白色
    # pcd.point_size = 2.0
    pcd.paint_uniform_color([0, 0, 0])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)

    # 绘制坐标轴
    # mesh_frame = o3d.open3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=10, origin=[-2, -2, -2])
    # vis.add_geometry(mesh_frame)
    # 绘制线条
    # polygon_points = np.array([[75, 75, 0], [-75, 75, 0], [0, 0, 0], [-75, 0, 0], [75, 0, 0]])
    # lines = [[0, 1], [0, 4], [1, 3], [0, 2], [1, 2], ]  # 连接的顺序，封闭链接
    # color = [[154/255, 205/255, 50/255] for i in range(len(lines)-2)]
    # color.append([255/255, 255/255, 255/255])
    # color.append([255/255, 255/255, 255/255])
    # lines_pcd = o3d.open3d.geometry.LineSet()
    # lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    # lines_pcd.lines = o3d.open3d.utility.Vector2iVector(lines)
    # lines_pcd.colors = o3d.open3d.utility.Vector3dVector(color)
    # vis.add_geometry(lines_pcd)



    h = 0
    # 绘制网格
    a = -75
    b = 75
    grid_points = np.array([[a, 0, 0-h], [a, b, 0-h],
                            [a, 0, 10-h], [a, b, 10-h],
                            [a, 0, 5-h], [a, b, 5-h],
                            [a, 0, 15-h], [a, b, 15-h],
                            [a, 0, 0-h], [a, 0, 15-h],
                            [a, b, 0-h], [a, b, 15-h],
                            [a, b/5*1, 0-h], [a, b/5*1, 15-h],
                            [a, b/5*2, 0-h], [a, b/5*2, 15-h],
                            [a, b/5*3, 0-h], [a, b/5*3, 15-h],
                            [a, b/5*4, 0-h], [a, b/5*4, 15-h],

                            ])
    lines = [[0, 1],[2, 3],[4,5],[6,7],[8,9],[10,11],
             [12, 13], [14, 15], [16, 17], [18, 19]]  # 连接的顺序，封闭链接
    color = [[192 / 255, 192 / 255, 192 / 255] for i in range(len(lines))]
    lines_pcd = o3d.open3d.geometry.LineSet()
    lines_pcd.points = o3d.utility.Vector3dVector(grid_points)
    lines_pcd.lines = o3d.open3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.open3d.utility.Vector3dVector(color)
    vis.add_geometry(lines_pcd)

    # # 绘制网格(右边界)
    # a = 75
    # b = 75
    # grid_points = np.array([[a, 0, 0], [a, b, 0],
    #                         [a, 0, 10], [a, b, 10],
    #                         [a, 0, 5], [a, b, 5],
    #                         [a, 0, 15], [a, b, 15],
    #                         [a, 0, 0], [a, 0, 15],
    #                         [a, b, 0], [a, b, 15],
    #                         [a, b / 5 * 1, 0], [a, b / 5 * 1, 15],
    #                         [a, b / 5 * 2, 0], [a, b / 5 * 2, 15],
    #                         [a, b / 5 * 3, 0], [a, b / 5 * 3, 15],
    #                         [a, b / 5 * 4, 0], [a, b / 5 * 4, 15],
    #
    #                         ])
    # lines = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
    #          [12, 13], [14, 15], [16, 17], [18, 19]]  # 连接的顺序，封闭链接
    # color = [[192 / 255, 192 / 255, 192 / 255] for i in range(len(lines))]
    # lines_pcd = o3d.open3d.geometry.LineSet()
    # lines_pcd.points = o3d.utility.Vector3dVector(grid_points)
    # lines_pcd.lines = o3d.open3d.utility.Vector2iVector(lines)
    # lines_pcd.colors = o3d.open3d.utility.Vector3dVector(color)
    # vis.add_geometry(lines_pcd)

    # 绘制网格(前边界)
    a = -75
    b = 75
    grid_points = np.array([[a, b, 0-h], [-a, b, 0-h],
                            [a, b, 10-h], [-a, b, 10-h],
                            [a, b, 5-h], [-a, b, 5-h],
                            [a, b, 15-h], [-a, b, 15-h],
                            [a, b, 0-h], [-a, b, 0-h],
                            [a, 0, 0-h], [-a, 0, 0-h],
                            [a, b / 5 * 1, 0-h], [-a, b / 5 * 1, 0-h],
                            [a, b / 5 * 2, 0-h], [-a, b / 5 * 2, 0-h],
                            [a, b / 5 * 3, 0-h], [-a, b / 5 * 3, 0-h],
                            [a, b / 5 * 4, 0-h], [-a, b / 5 * 4, 0-h],

                            ])
    lines = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
             [12, 13], [14, 15], [16, 17], [18, 19]]  # 连接的顺序，封闭链接
    color = [[192 / 255, 192 / 255, 192 / 255] for i in range(len(lines))]
    lines_pcd = o3d.open3d.geometry.LineSet()
    lines_pcd.points = o3d.utility.Vector3dVector(grid_points)
    lines_pcd.lines = o3d.open3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.open3d.utility.Vector3dVector(color)
    vis.add_geometry(lines_pcd)

    # 绘制网格(di边界)
    a = -75
    b = 75
    grid_points = np.array([
                            [a/5*1, 0, 0-h], [a/5*1, b, 0-h],
                            [a/5*2, 0, 0-h], [a/5*2, b, 0-h],
                            [a/5*3, 0, 0-h], [a/5*3, b, 0-h],
                            [a/5*4, 0, 0-h], [a/5*4, b, 0-h],
                            [0, 0, 0-h], [0, b, 0-h],
                            [-a / 5 * 1, 0, 0-h], [-a / 5 * 1, b, 0-h],
                            [-a / 5 * 2, 0, 0-h], [-a / 5 * 2, b, 0-h],
                            [-a / 5 * 3, 0, 0-h], [-a / 5 * 3, b, 0-h],
                            [-a / 5 * 4, 0, 0-h], [-a / 5 * 4, b, 0-h],
                            [-a / 5 * 5, 0, 0-h], [-a / 5 * 5, b, 0-h],
                            ])
    lines = [[0, 1], [2, 3], [4, 5], [6, 7],[8,9],[10, 11],
             [12, 13], [14, 15], [16, 17], [18,19]]  # 连接的顺序，封闭链接
    color = [[192 / 255, 192 / 255, 192 / 255] for i in range(len(lines))]
    lines_pcd = o3d.open3d.geometry.LineSet()
    lines_pcd.points = o3d.utility.Vector3dVector(grid_points)
    lines_pcd.lines = o3d.open3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.open3d.utility.Vector3dVector(color)
    vis.add_geometry(lines_pcd)

    # 绘制网格(前边界)
    a = -75
    b = 75
    grid_points = np.array([
        [a / 5 * 1, b, 0-h], [a / 5 * 1, b, 15-h],
        [a / 5 * 2, b, 0-h], [a / 5 * 2, b, 15-h],
        [a / 5 * 3, b, 0-h], [a / 5 * 3, b, 15-h],
        [a / 5 * 4, b, 0-h], [a / 5 * 4, b, 15-h],
        [0, b, 0-h], [0, b, 15-h],
        [-a / 5 * 1, b, 0-h], [-a / 5 * 1, b, 15-h],
        [-a / 5 * 2, b, 0-h], [-a / 5 * 2, b, 15-h],
        [-a / 5 * 3, b, 0-h], [-a / 5 * 3, b, 15-h],
        [-a / 5 * 4, b, 0-h], [-a / 5 * 4, b, 15-h],
        [-a / 5 * 5, b, 0-h], [-a / 5 * 5, b, 15-h],
    ])
    lines = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
             [12, 13], [14, 15], [16, 17], [18, 19]]  # 连接的顺序，封闭链接
    color = [[192 / 255, 192 / 255, 192 / 255] for i in range(len(lines))]
    lines_pcd = o3d.open3d.geometry.LineSet()
    lines_pcd.points = o3d.utility.Vector3dVector(grid_points)
    lines_pcd.lines = o3d.open3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.open3d.utility.Vector3dVector(color)
    vis.add_geometry(lines_pcd)


    # FOV
    grid_points = np.array([
        [0, 0, 0-h], [-75, 75, 0-h],[75, 75, 0-h]
    ])
    lines = [[0, 1], [0, 2]]  # 连接的顺序，封闭链接
    color = [[255 / 255, 0 / 255, 0 / 255] for i in range(len(lines))]
    lines_pcd = o3d.open3d.geometry.LineSet()
    lines_pcd.points = o3d.utility.Vector3dVector(grid_points)
    lines_pcd.lines = o3d.open3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.open3d.utility.Vector3dVector(color)
    vis.add_geometry(lines_pcd)

    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=5)

    vis.run()
    vis.capture_screen_image("save.jpg")
    vis.destroy_window()


if __name__ == "__main__":
    main()
    img = Image.open('save.jpg')
    img = np.asarray(img)
    h, w = img.shape[:2]
    im = img[h//4:h//4*3, w//4:w//4*3]
    Image.fromarray(im).save('radar.jpg')
