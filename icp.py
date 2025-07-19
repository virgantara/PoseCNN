import open3d as o3d
import numpy as np

def refine_pose_with_icp(model_points, depth_points, init_pose, threshold=0.02, max_iter=50):
    """
    Refines pose using ICP between model points and observed depth points.
    """
    # Convert to Open3D PointClouds
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(model_points)
    target.points = o3d.utility.Vector3dVector(depth_points)

    # Apply initial transform
    source.transform(init_pose)

    # ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.identity(4),  # identity (initial was already applied above)
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )

    return reg_p2p.transformation
