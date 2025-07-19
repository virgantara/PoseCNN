import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from sklearn.metrics import auc

def compute_auc(scores, thresholds=np.linspace(0, 0.1, 100)):
    """
    Compute the AUC (Area Under Curve) for pose accuracy vs threshold.

    Args:
        scores (list of float): ADD or ADD-S distances in meters.
        thresholds (np.ndarray): Distance thresholds (default 0 to 10cm).

    Returns:
        auc_value (float): Area under the curve (normalized).
    """
    scores = np.array(scores)
    accuracies = [(scores < t).mean() for t in thresholds]
    auc_value = auc(thresholds, accuracies) / thresholds[-1]  # Normalize to [0, 1]
    return auc_value

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

def compute_adds(R_gt, t_gt, R_pred, t_pred, model_points):
    """
    ADD-S: Average Distance for symmetric objects
    """
    pts_gt = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)
    pts_pred = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)

    tree = cKDTree(pts_pred)
    distances, _ = tree.query(pts_gt, k=1)  # Closest point distance
    return np.mean(distances)

