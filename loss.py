import torch
import torch.nn as nn

def get_chamfer_distance(cloud1, cloud2):
    if cloud1.ndim == 2: cloud1 = cloud1.unsqueeze(0)
    if cloud2.ndim == 2: cloud2 = cloud2.unsqueeze(0)
    
    dist_matrix = torch.cdist(cloud1, cloud2)
    min_dist_p1, _ = torch.min(dist_matrix, dim=2)
    min_dist_p2, _ = torch.min(dist_matrix, dim=1)
    return torch.mean(min_dist_p1, dim=1) + torch.mean(min_dist_p2, dim=1)

def weighted_chamfer_distance(cloud1, cloud2, w_p1_to_p2=1.0, w_p2_to_p1=1.0):
    if cloud1.ndim == 2: cloud1 = cloud1.unsqueeze(0)
    if cloud2.ndim == 2: cloud2 = cloud2.unsqueeze(0)
    
    dist_matrix = torch.cdist(cloud1, cloud2)
    min_dist_p1, _ = torch.min(dist_matrix, dim=2)
    min_dist_p2, _ = torch.min(dist_matrix, dim=1)
    return w_p1_to_p2 * torch.mean(min_dist_p1, dim=1) + w_p2_to_p1 * torch.mean(min_dist_p2, dim=1)

def get_hausdorff_distance(cloud1, cloud2):
    if cloud1.ndim == 2: cloud1 = cloud1.unsqueeze(0)
    if cloud2.ndim == 2: cloud2 = cloud2.unsqueeze(0)
    
    dist_matrix = torch.cdist(cloud1, cloud2)
    min_dist_1_to_2, _ = torch.min(dist_matrix, dim=2)
    min_dist_2_to_1, _ = torch.min(dist_matrix, dim=1)
    h_12 = torch.max(min_dist_1_to_2, dim=1).values
    h_21 = torch.max(min_dist_2_to_1, dim=1).values
    return h_12 + h_21
    # return torch.max(torch.stack([h_12, h_21], dim=1), dim=1).values

def align_pca(p):
    if p.ndim == 2:
        p = p.unsqueeze(0)
    centroid = torch.mean(p, dim=1, keepdim=True)
    p_centered = p - centroid
    cov = torch.bmm(p_centered.transpose(1, 2), p_centered)
    e, v = torch.linalg.eigh(cov)
    p_aligned = torch.bmm(p_centered, v)
    return p_aligned

def invariant_chamfer_loss(cloud1, cloud2, return_mean=False):
    if cloud1.ndim == 2: cloud1 = cloud1.unsqueeze(0)
    if cloud2.ndim == 2: cloud2 = cloud2.unsqueeze(0)
    c1_aligned = align_pca(cloud1)
    c2_aligned = align_pca(cloud2)
    base_loss = get_chamfer_distance(c1_aligned, c2_aligned)
    mirror_x = torch.tensor([-1.0, 1.0, 1.0], device=cloud1.device)
    c2_flipped_x = c2_aligned * mirror_x
    loss_x_flip = get_chamfer_distance(c1_aligned, c2_flipped_x)
    mirror_y = torch.tensor([1.0, -1.0, 1.0], device=cloud1.device)
    c2_flipped_y = c2_aligned * mirror_y
    loss_y_flip = get_chamfer_distance(c1_aligned, c2_flipped_y)
    c2_flipped_xy = c2_aligned * mirror_x * mirror_y
    loss_xy_flip = get_chamfer_distance(c1_aligned, c2_flipped_xy)
    losses = torch.stack([base_loss, loss_x_flip, loss_y_flip, loss_xy_flip], dim=0)
    min_loss, _ = torch.min(losses, dim=0)
    if return_mean:
        return torch.mean(min_loss)
    else:
        return min_loss

def invariant_hausdorff_loss(cloud1, cloud2, return_mean=False):
    if cloud1.ndim == 2: cloud1 = cloud1.unsqueeze(0)
    if cloud2.ndim == 2: cloud2 = cloud2.unsqueeze(0)
    c1_aligned = align_pca(cloud1)
    c2_aligned = align_pca(cloud2)
    base_loss = get_hausdorff_distance(c1_aligned, c2_aligned)
    mirror_x = torch.tensor([-1.0, 1.0, 1.0], device=cloud1.device)
    c2_flipped_x = c2_aligned * mirror_x
    loss_x_flip = get_hausdorff_distance(c1_aligned, c2_flipped_x)
    mirror_y = torch.tensor([1.0, -1.0, 1.0], device=cloud1.device)
    c2_flipped_y = c2_aligned * mirror_y
    loss_y_flip = get_hausdorff_distance(c1_aligned, c2_flipped_y)
    c2_flipped_xy = c2_aligned * mirror_x * mirror_y
    loss_xy_flip = get_hausdorff_distance(c1_aligned, c2_flipped_xy)
    losses = torch.stack([base_loss, loss_x_flip, loss_y_flip, loss_xy_flip], dim=0)
    min_loss, _ = torch.min(losses, dim=0)
    if return_mean:
        return torch.mean(min_loss)
    else:
        return min_loss

def normalize_pointcloud(points):
    """
    Normalize point cloud to canonical space:
    - Center at origin
    - Scale to unit variance
    
    Args:
        points: (B, N, 3) or (N, 3) tensor
    Returns:
        Normalized points with same shape
    """
    if points.ndim == 2:
        points = points.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Center at origin
    centroid = torch.mean(points, dim=1, keepdim=True)
    points_centered = points - centroid
    
    # Scale to unit variance (more stable than bounding box for gradient flow)
    scale = torch.sqrt(torch.mean(points_centered ** 2, dim=(1, 2), keepdim=True)) + 1e-8
    points_normalized = points_centered / scale
    
    if squeeze_output:
        return points_normalized.squeeze(0)
    return points_normalized

def procrustes_align(source, target):
    """
    Align source point cloud to target using Procrustes analysis.
    Assumes both clouds are already centered and scaled.
    
    Args:
        source: (B, N_source, 3) tensor - cloud to be aligned
        target: (B, N_target, 3) tensor - reference cloud
    Returns:
        Aligned source cloud (B, N_source, 3)
    """
    if source.ndim == 2:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = source.shape[0]
    aligned_sources = []
    
    for b in range(batch_size):
        src_b = source[b]  # (N_source, 3)
        tgt_b = target[b]  # (N_target, 3)
        
        # Step 1: Find nearest neighbors to establish correspondences
        dist_matrix = torch.cdist(src_b.unsqueeze(0), tgt_b.unsqueeze(0)).squeeze(0)  # (N_source, N_target)
        nn_indices = torch.argmin(dist_matrix, dim=1)  # (N_source,)
        target_matched = tgt_b[nn_indices]  # (N_source, 3)
        
        # Step 2: Compute optimal rotation using SVD
        # H = source^T @ target_matched
        H = src_b.T @ target_matched  # (3, 3)
        
        # SVD: H = U @ S @ V^T
        U, S, Vt = torch.linalg.svd(H)
        V = Vt.T
        
        # Compute rotation matrix: R = V @ U^T
        R = V @ U.T  # (3, 3)
        
        # Handle reflection case (det(R) = -1)
        # Ensure proper rotation (det(R) = +1)
        if torch.det(R) < 0:
            V_new = V.clone()
            V_new[:, -1] *= -1
            R = V_new @ U.T
        
        # Step 3: Apply rotation to source
        aligned_src = src_b @ R  # (N_source, 3)
        aligned_sources.append(aligned_src)
    
    result = torch.stack(aligned_sources, dim=0)  # (B, N_source, 3)
    
    if squeeze_output:
        return result.squeeze(0)
    return result

def invariant_chamfer_loss_2(cloud1, cloud2, return_mean=False):
    """
    Translation, scale, and rotation invariant Chamfer distance loss.
    Uses normalization + Procrustes alignment + Chamfer distance.
    
    This hybrid approach:
    1. Normalizes both clouds (translation + scale invariance)
    2. Aligns cloud1 to cloud2 using one iteration of Procrustes (rotation invariance)
    3. Computes Chamfer distance on aligned clouds
    
    Args:
        cloud1: (B, N1, 3) or (N1, 3) predicted point cloud
        cloud2: (B, N2, 3) or (N2, 3) target point cloud
        return_mean: If True, return scalar mean. If False, return per-batch losses.
    
    Returns:
        Chamfer distance loss (scalar if return_mean=True, else (B,) tensor)
    """
    if cloud1.ndim == 2:
        cloud1 = cloud1.unsqueeze(0)
    if cloud2.ndim == 2:
        cloud2 = cloud2.unsqueeze(0)
    
    # Step 1: Normalize both clouds (center + scale)
    cloud1_norm = normalize_pointcloud(cloud1)
    cloud2_norm = normalize_pointcloud(cloud2)
    
    # Step 2: Align cloud1 to cloud2 using Procrustes
    cloud1_aligned = procrustes_align(cloud1_norm, cloud2_norm)
    
    # Step 3: Compute Chamfer distance on aligned clouds
    loss = get_chamfer_distance(cloud1_aligned, cloud2_norm)
    
    if return_mean:
        return torch.mean(loss)
    else:
        return loss

class InvariantChamferLoss(nn.Module):
    def __init__(self, num_iters=5, min_scale=0.5, max_scale=2.0, eps=1e-8):
        """
        Args:
            num_iters: Number of ICP iterations to align point clouds.
            min_scale: Minimum allowed scale factor (prevents singularity collapse).
            max_scale: Maximum allowed scale factor (prevents extreme divergence).
            eps: Small value to prevent division by zero.
        """
        super().__init__()
        self.num_iters = num_iters
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.eps = eps

    def get_umeyama_transform(self, source, target_corr):
        """
        Computes the optimal rigid transformation + scale. 
        Expected to be called inside torch.no_grad().
        """
        B, N, dim = source.shape

        # 1. Compute Centroids
        mu_src = source.mean(dim=1, keepdim=True)
        mu_tgt = target_corr.mean(dim=1, keepdim=True)

        # 2. Center the point clouds
        src_c = source - mu_src
        tgt_c = target_corr - mu_tgt

        # 3. Covariance Matrix
        H = torch.bmm(src_c.transpose(1, 2), tgt_c)

        # 4. Singular Value Decomposition
        U, S, Vh = torch.linalg.svd(H)
        V = Vh.transpose(1, 2)

        # 5. Handle Reflections
        det_UV = torch.det(torch.bmm(V, U.transpose(1, 2)))
        R_det = torch.eye(dim, device=source.device, dtype=source.dtype).unsqueeze(0).repeat(B, 1, 1)
        R_det[:, -1, -1] = det_UV 

        # 6. Optimal Rotation
        R = torch.bmm(torch.bmm(V, R_det), U.transpose(1, 2))

        # 7. Optimal Scale with Clamping
        var_src = (src_c ** 2).sum(dim=(1, 2)) 
        trace_S = (S * torch.diagonal(R_det, dim1=1, dim2=2)).sum(dim=1)
        scale = (trace_S / (var_src + self.eps)).unsqueeze(1).unsqueeze(2)
        
        # Restrict scale to avoid singularity collapse
        scale = torch.clamp(scale, min=self.min_scale, max=self.max_scale)

        # 8. Optimal Translation (Must be computed AFTER clamping scale)
        t = mu_tgt - scale * torch.bmm(mu_src, R.transpose(1, 2))

        return R, t, scale

    def forward(self, source, target, return_mean=True):
        """
        source: (B, N, 3) - Requires Gradients
        target: (B, M, 3) - Detached / Ground Truth
        """
        # ---------------------------------------------------------
        # Phase 1: Detached Alignment (No Gradients)
        # ---------------------------------------------------------
        with torch.no_grad():
            src_current = source.clone()

            for _ in range(self.num_iters):
                # Find nearest neighbors
                dist_matrix = torch.cdist(src_current, target, p=2.0) ** 2
                min_indices = torch.argmin(dist_matrix, dim=2)
                
                # Gather corresponding target points
                idx_expanded = min_indices.unsqueeze(-1).expand(-1, -1, 3)
                tgt_corr = torch.gather(target, 1, idx_expanded)
                
                # Compute transform from ORIGINAL source to current correspondences.
                # This guarantees the scale calculation represents the absolute scale.
                R, t, scale = self.get_umeyama_transform(source, tgt_corr)
                
                # Update temporary source for the next ICP iteration
                src_current = scale * torch.bmm(source, R.transpose(1, 2)) + t

        # ---------------------------------------------------------
        # Phase 2: Differentiable Loss Calculation
        # ---------------------------------------------------------
        # Apply the final, detached transformation parameters to the ATTACHED source.
        # Gradients will flow through `source` but bypass the SVD entirely.
        aligned_source = scale * torch.bmm(source, R.transpose(1, 2)) + t

        # Final Bidirectional Chamfer Distance
        final_dist = torch.cdist(aligned_source, target, p=2.0) ** 2
        
        min_src2tgt = torch.min(final_dist, dim=2)[0]
        min_tgt2src = torch.min(final_dist, dim=1)[0]
        
        chamfer_loss = min_src2tgt.mean(dim=1) + min_tgt2src.mean(dim=1)
        
        if return_mean:
            return chamfer_loss.mean(), aligned_source
        else:
            return chamfer_loss, aligned_source