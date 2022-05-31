import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


data = np.load('/home/wilson/repos/lab/trajectory/data.npz')
T = data['depth'].shape[0]

all_points = []
all_colors = []
for t in range(T):
    p_matrix = data['projection_matrix'][t]
    mv_matrix = data['modelview_matrix'][t]
    depth_frame = data['depth'][t]
    rgb_frame = data['rgb'][t] / 255.

    x = y = np.linspace(0, 1, depth_frame.shape[0])
    coords = np.stack(np.meshgrid(x, y), axis=-1)
    coords = np.reshape(coords, (-1, 2))

    coords = 2 * coords - 1
    z = np.reshape(2 * depth_frame - 1, (-1, 1))

    clip_points = np.concatenate([coords, z, np.ones_like(z)], axis=-1)

    point = (np.linalg.inv(p_matrix) @ clip_points.T).T
    point = (np.linalg.inv(mv_matrix) @ point.T).T
    point /= point[:, [-1]]
    point = point[:, :-1]

    colors = np.reshape(rgb_frame, (-1, 3))

    all_points.append(point)
    all_colors.append(colors)

all_points = np.concatenate(all_points)
all_colors = np.concatenate(all_colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(all_colors)

o3d.visualization.draw_geometries([pcd])
