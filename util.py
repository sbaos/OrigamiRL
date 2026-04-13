import torch

def pointcloud_sampling_centroid_no_unique(points, faces, lines):
    points_sampling = (points[:, faces[:, 0]] + points[:, faces[:, 2]] + points[:, faces[:, 3]]) / 3.0
    points_sampling2 = (points[:, faces[:, 1]] + points[:, faces[:, 2]] + points[:, faces[:, 3]]) / 3.0
    points_sampling3 = (points[:, lines[:, 0]] + points[:, lines[:, 1]])
    return torch.cat([points, points_sampling, points_sampling2, points_sampling3], dim=1)

def pointcloud_sampling_centroid(points, faces_indices_unique):
    p0 = points[:, faces_indices_unique[:, 0]]
    p1 = points[:, faces_indices_unique[:, 1]]
    p2 = points[:, faces_indices_unique[:, 2]]
    # 1st subdivision
    mp0p1 = (p0 + p1) / 2.0
    mp1p2 = (p1 + p2) / 2.0
    mp0p2 = (p0 + p2) / 2.0
    # centroid
    g = (p0 + p1 + p2) / 3.0

    points_sampling = torch.cat([
        points, mp0p1, mp1p2, mp0p2, g
    ], dim=1)
    return points_sampling

def pointcloud_sampling(points, faces_indices_unique):
    p0 = points[:, faces_indices_unique[:, 0]]
    p1 = points[:, faces_indices_unique[:, 1]]
    p2 = points[:, faces_indices_unique[:, 2]]
    # 1st subdivision
    mp0p1 = (p0 + p1) / 2.0
    mp1p2 = (p1 + p2) / 2.0
    mp0p2 = (p0 + p2) / 2.0
    # 2nd subdivision
    p0_mp0p1 = (p0 + mp0p1) / 2.0
    p1_mp0p1 = (p1 + mp0p1) / 2.0
    p1_mp1p2 = (p1 + mp1p2) / 2.0
    p2_mp1p2 = (p2 + mp1p2) / 2.0
    p0_mp0p2 = (p0 + mp0p2) / 2.0
    p2_mp0p2 = (p2 + mp0p2) / 2.0
    mmp0p1mp1p2 = (mp0p1 + mp1p2) / 2.0
    mmp0p2mp1p2 = (mp0p2 + mp1p2) / 2.0
    mmp0p1mp0p2 = (mp0p1 + mp0p2) / 2.0

    points_sampling = torch.cat([
        points, mp0p1, mp1p2, mp0p2, p0_mp0p1, p1_mp0p1, p1_mp1p2, p2_mp1p2, p0_mp0p2, p2_mp0p2, mmp0p1mp1p2, mmp0p2mp1p2, mmp0p1mp0p2
    ], dim=1)
    return points_sampling

def pointcloud_sampling2(points, faces_indices_unique, num_subdivisions=3):
    device = points.device
    p0 = points[:, faces_indices_unique[:, 0]] # (B, F, 3)
    p1 = points[:, faces_indices_unique[:, 1]]
    p2 = points[:, faces_indices_unique[:, 2]]
    segments = 2 ** num_subdivisions
    steps = torch.arange(segments + 1, device=device, dtype=points.dtype)
    u_idx, v_idx = torch.meshgrid(steps, steps, indexing='ij')
    mask = (u_idx + v_idx) <= segments
    u = u_idx[mask] / segments
    v = v_idx[mask] / segments
    w = 1.0 - u - v
    # Reshape for broadcasting: (1, 1, M, 1)
    u = u.view(1, 1, -1, 1)
    v = v.view(1, 1, -1, 1)
    w = w.view(1, 1, -1, 1)
    # Expand points: (B, F, 1, 3)
    p0 = p0.unsqueeze(2)
    p1 = p1.unsqueeze(2)
    p2 = p2.unsqueeze(2)
    # (B, F, M, 3)
    points_sampling = u * p0 + v * p1 + w * p2
    # Flatten
    B, F, M, D = points_sampling.shape
    points_sampling = points_sampling.view(B, F * M, D)
    return points_sampling

def pointcloud_sampling3(points, faces_indices_unique, num_random_points=16):
    device = points.device
    dtype = points.dtype
    B = points.shape[0]
    F = faces_indices_unique.shape[0]
    
    p0 = points[:, faces_indices_unique[:, 0]] # (B, F, 3)
    p1 = points[:, faces_indices_unique[:, 1]]
    p2 = points[:, faces_indices_unique[:, 2]]
    
    # Generate random barycentric coordinates
    # We generate independent random points for each face in each batch to avoid patterns
    # r1, r2 ~ U[0, 1]
    r1 = torch.rand((B, F, num_random_points, 1), device=device, dtype=dtype)
    r2 = torch.rand((B, F, num_random_points, 1), device=device, dtype=dtype)
    
    # Uniform sampling on triangle:
    # u = 1 - sqrt(r1)
    # v = sqrt(r1) * (1 - r2)
    # w = sqrt(r1) * r2
    sqrt_r1 = torch.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    
    # Expand points: (B, F, 1, 3)
    p0 = p0.unsqueeze(2)
    p1 = p1.unsqueeze(2)
    p2 = p2.unsqueeze(2)
    
    # (B, F, M, 3)
    points_sampling = u * p0 + v * p1 + w * p2
    
    # Flatten
    points_sampling = points_sampling.view(B, F * num_random_points, 3)
    return points_sampling

def pointcloud_min_max_normalize(points):
    # Normalize to [-1, 1]
    min_vals = torch.min(points)
    max_vals = torch.max(points)
    points_normalized = (points - min_vals) / (max_vals - min_vals)
    points_normalized = points_normalized * 2 - 1
    return points_normalized


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # points = torch.tensor([
    #     [-1., 0., 0.], [0.3, -2., 0.], [1., 0., 0.], [-0.5, 1.5, 0.],
    # ]).unsqueeze(0)
    points = torch.tensor([
        [-1., 0., 0.], [0., 2., 0.], [1., 0., 0.],
    ]).unsqueeze(0)

    faces_indices_unique = torch.tensor([
        [0, 1, 2],
    ])
    plt.scatter(points[0, :, 0], points[0, :, 1])
    plt.show()
    points_sampling = pointcloud_sampling2(points, faces_indices_unique, num_subdivisions=3)
    plt.scatter(points_sampling[0, :, 0], points_sampling[0, :, 1], label='Subdivision')
    plt.show()
    
    # Test pointcloud_sampling3
    points_sampling3 = pointcloud_sampling3(points, faces_indices_unique, num_random_points=160)
    plt.scatter(points_sampling3[0, :, 0], points_sampling3[0, :, 1], label='Random Sampling')
    plt.legend()
    plt.show()

    points_sampling_centroid = pointcloud_sampling_centroid(points, faces_indices_unique)
    plt.scatter(points_sampling_centroid[0, :, 0], points_sampling_centroid[0, :, 1], label='Centroid')
    plt.legend()
    plt.show()