import numpy as np

def quaternion_to_rotation_matrix(q):
    # Assumes q is [x, y, z, w]
    x, y, z, w = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])
    return R
    
def compute_add(R_gt, t_gt, R_pred, t_pred, model_points):
    """
    Compute the ADD metric for 6D pose estimation.
    
    Parameters:
        R_gt (np.ndarray): Ground truth rotation (3x3)
        t_gt (np.ndarray): Ground truth translation (3,)
        R_pred (np.ndarray): Predicted rotation (3x3)
        t_pred (np.ndarray): Predicted translation (3,)
        model_points (np.ndarray): Object model points (Nx3)
        
    Returns:
        float: Mean Euclidean distance (ADD)
    """
    pts_gt = (R_gt @ model_points.T).T + t_gt
    pts_pred = (R_pred @ model_points.T).T + t_pred
    return np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1))