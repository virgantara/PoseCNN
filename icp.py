import open3d as o3d
import numpy as np

def refine_pose_with_icp(src_points, tgt_points, init_pose=np.eye(4), threshold=0.02, max_iter=50):
    # Ensure float64 and CPU-based pcds
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(np.asarray(src_points).astype(np.float64))
    tgt_pcd.points = o3d.utility.Vector3dVector(np.asarray(tgt_points).astype(np.float64))

    init_pose = init_pose.astype(np.float64)  # Ensure same dtype

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=src_pcd,
        target=tgt_pcd,
        max_correspondence_distance=threshold,
        init=init_pose,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )

    return reg_p2p.transformation
