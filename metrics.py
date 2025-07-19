import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

def compute_add(R_gt, t_gt, R_pred, t_pred, model_points):
    pts_gt = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)
    pts_pred = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)
    return np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1))