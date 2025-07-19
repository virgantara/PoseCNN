import open3d as o3d
import numpy as np

def refine_pose_with_icp(src_points, tgt_points, init_pose=np.eye(4), threshold=0.02, max_iter=50):
    # Convert to Open3D point cloud
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_points)
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)

    # ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src=src_pcd,
        target=tgt_pcd,
        max_correspondence_distance=threshold,
        init=init_pose,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )

    return reg_p2p.transformation
