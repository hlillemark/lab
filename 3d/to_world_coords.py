import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.ndimage import zoom


def compute_mv_matrix(eye, angle):
    rot = Rotation.from_quat(angle).as_matrix()
    mv_matrix = np.concatenate([rot, eye[:, None]], axis=-1)
    mv_matrix = np.concatenate([mv_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    return mv_matrix


R = 128
data = np.load(sys.argv[1])
T = data['video'].shape[0]

all_points = []
all_colors = []
for t in range(T):
    eye = data['pos'][t]
    angle = data['rot'][t]
    mv_matrix = compute_mv_matrix(eye, angle)
    
    p_matrix = data['proj_matrices'][t]
    depth_frame = data['depth_video'][t]
    F = R / depth_frame.shape[0]
    depth_frame = zoom(depth_frame, (F, F, 1), order=1)
    rgb_frame = data['video'][t] / 255.
    rgb_frame = zoom(rgb_frame, (F, F, 1), order=1)

    x = y = np.linspace(0, 1, R)
    coords = np.stack(np.meshgrid(x, y), axis=-1)
    coords = np.reshape(coords, (-1, 2))

    coords = 2 * coords - 1
    z = np.reshape(2 * depth_frame - 1, (-1, 1))

    clip_points = np.concatenate([coords, z, np.ones_like(z)], axis=-1)

    point = (np.linalg.inv(p_matrix) @ clip_points.T).T
    point = (mv_matrix @ point.T).T
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
