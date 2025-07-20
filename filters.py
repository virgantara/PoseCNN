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

    # print(f"ICP fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")

    return reg_p2p.transformation, reg_p2p.fitness


def refine_pose_with_guided_filter(src_points, tgt_points, init_pose=np.eye(4), radius=0.05, epsilon=1e-6):
    # Ensure float64 and CPU-based point clouds
    src_points = np.asarray(src_points).astype(np.float64)
    tgt_points = np.asarray(tgt_points).astype(np.float64)

    # Apply initial pose to source
    src_points_h = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
    src_transformed = (init_pose @ src_points_h.T).T[:, :3]

    # Create point clouds
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_transformed)
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)

    # Guided filter pose refinement
    kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)
    refined_src = np.copy(src_transformed)
    valid_pairs = []

    for i in range(len(src_transformed)):
        k, idx, _ = kdtree.search_radius_vector_3d(src_transformed[i], radius)
        if k < 3:
            continue

        tgt_neighbors = tgt_points[idx]
        mean_tgt = np.mean(tgt_neighbors, axis=0)
        cov_tgt = np.cov(tgt_neighbors.T)
        e = np.linalg.inv(cov_tgt + epsilon * np.eye(3))

        A = cov_tgt @ e
        b = mean_tgt - A @ mean_tgt

        guided_point = A @ src_transformed[i] + b
        refined_src[i] = guided_point
        valid_pairs.append((guided_point, mean_tgt))

    if len(valid_pairs) < 3:
        return np.eye(4), 0.0

    refined_pts, tgt_pts = zip(*valid_pairs)
    refined_pts = np.stack(refined_pts)
    tgt_pts = np.stack(tgt_pts)

    # SVD for best-fit transform
    src_centroid = np.mean(refined_pts, axis=0)
    tgt_centroid = np.mean(tgt_pts, axis=0)
    src_centered = refined_pts - src_centroid
    tgt_centered = tgt_pts - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # Fitness calculation (like Open3D ICP fitness)
    distances = np.linalg.norm((R @ refined_pts.T).T + t - tgt_pts, axis=1)
    threshold = radius
    inliers = distances < threshold
    fitness = np.sum(inliers) / len(src_points)

    return T @ init_pose, fitness

def refine_pose_with_guided_filter_improved(
    src_points, tgt_points, init_pose=np.eye(4),
    radius=0.05, epsilon=1e-5, max_iter=10, mahal_thresh=3.0
):
    src_points = np.asarray(src_points).astype(np.float64)
    tgt_points = np.asarray(tgt_points).astype(np.float64)

    # Apply initial transformation
    T_total = init_pose.copy()
    src_points_h = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
    src_transformed = (T_total @ src_points_h.T).T[:, :3]

    for _ in range(max_iter):
        src_pcd = o3d.geometry.PointCloud()
        tgt_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_transformed)
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)
        kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)

        refined_src = []
        refined_tgt = []
        weights = []

        for i in range(len(src_transformed)):
            k, idx, _ = kdtree.search_radius_vector_3d(src_transformed[i], radius)
            if k < 3:
                continue

            neighbors = tgt_points[idx]
            mean_tgt = np.mean(neighbors, axis=0)
            cov_tgt = np.cov(neighbors.T) + epsilon * np.eye(3)

            diff = src_transformed[i] - mean_tgt
            try:
                cov_inv = np.linalg.inv(cov_tgt)
            except np.linalg.LinAlgError:
                continue

            maha_dist = np.sqrt(diff @ cov_inv @ diff)
            if maha_dist > mahal_thresh:
                continue

            # Guided filter mapping
            A = cov_tgt @ cov_inv
            b = mean_tgt - A @ mean_tgt
            guided_point = A @ src_transformed[i] + b

            refined_src.append(guided_point)
            refined_tgt.append(mean_tgt)
            weights.append(np.exp(-maha_dist))  # Gaussian weighting

        if len(refined_src) < 3:
            break

        refined_src = np.array(refined_src)
        refined_tgt = np.array(refined_tgt)
        weights = np.array(weights).reshape(-1, 1)

        # Weighted centroids
        src_centroid = np.sum(refined_src * weights, axis=0) / np.sum(weights)
        tgt_centroid = np.sum(refined_tgt * weights, axis=0) / np.sum(weights)

        src_centered = refined_src - src_centroid
        tgt_centered = refined_tgt - tgt_centroid

        H = (weights * src_centered).T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid
        T_delta = np.eye(4)
        T_delta[:3, :3] = R
        T_delta[:3, 3] = t

        # Update total transformation
        T_total = T_delta @ T_total
        src_points_h = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
        src_transformed = (T_total @ src_points_h.T).T[:, :3]

    # Fitness
    final_distances = np.linalg.norm(refined_src - refined_tgt, axis=1)
    inliers = final_distances < radius
    fitness = np.sum(inliers) / len(src_points)

    return T_total, fitness
