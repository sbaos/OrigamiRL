import torch
import math
import jax
import jax.numpy as jnp
import numpy as np

class OrigamiObjectMatrix:
    def __init__(self, points: torch.Tensor,
                 lines: torch.Tensor,
                 faces: torch.Tensor,
                 target_theta: torch.Tensor,
                 mass: float = 1.0,
                 ea: float = 20.0,
                 k_crease: float = 0.7,
                 damping: float = 0.45,
                 fold_percent: float = 0.99,
                 dt: float = -1.0, # -1 means auto config 
                 min_force: float = 0.0001,
                 k_facet: float = 0.7,
                 use_projection: bool = True):
        self.use_projection = use_projection
        self.points = points
        self.lines = lines
        self.faces = faces

        self.num_points = points.shape[0]
        self.num_lines = lines.shape[0]
        self.num_faces = faces.shape[0]
        self.masses = mass
        self.ea = ea
        self.k_crease = torch.full_like(target_theta, k_crease)
        self.theta = torch.zeros_like(target_theta)
        self.target_theta = (target_theta*fold_percent)
        self.k_crease[torch.abs(self.target_theta) <= 0.01] = k_facet

        self.origin_length = torch.norm(self.points[:, self.faces[:, 2]] - self.points[:, self.faces[:, 3]], dim=2, keepdim=True)

        self.damping = damping
        self.velocities = torch.zeros_like(self.points)
        self.total_forces = torch.zeros_like(self.points)

        p1 = self.points[:, self.lines[:, 0]]
        p2 = self.points[:, self.lines[:, 1]]
        self.rest_lengths = torch.norm(p2 - p1, dim=2, keepdim=True)
        self.ea_rest_lengths = self.ea / self.rest_lengths

        self.dt = dt
        if dt < 0.0:
            self.dt = 1.0/(2.0*math.pi*math.sqrt(self.ea/self.rest_lengths.min().item()))
        self.line_idx0 = self.lines[:, 0]
        self.line_idx1 = self.lines[:, 1]
        self.face_idx0 = self.faces[:, 0]
        self.face_idx1 = self.faces[:, 1]
        self.face_idx2 = self.faces[:, 2]
        self.face_idx3 = self.faces[:, 3]
        self.min_force = min_force
        self.damping_coef = 2.0*self.damping*torch.sqrt(self.ea_rest_lengths)

    def step(self):
        self.total_forces = torch.zeros_like(self.points) # Reset for each step
        point1 = self.points[:, self.line_idx0]
        point2 = self.points[:, self.line_idx1]

        # fold force
        p1 = self.points[:, self.face_idx0]
        p2 = self.points[:, self.face_idx1]
        p3 = self.points[:, self.face_idx2]
        p4 = self.points[:, self.face_idx3]

        vec_p3p1 = p1 - p3
        vec_p3p2 = p2 - p3
        vec_p3p4 = p4 - p3
        vec_n1 = torch.cross(vec_p3p1, vec_p3p4, dim=2)
        vec_n2 = torch.cross(vec_p3p4, vec_p3p2, dim=2)

        lensq_n1 = (vec_n1**2).sum(dim=2, keepdim=True)
        lensq_n2 = (vec_n2**2).sum(dim=2, keepdim=True)
        lensq_n1 = torch.clamp(lensq_n1, min=1e-6)
        lensq_n2 = torch.clamp(lensq_n2, min=1e-6)
        
        len_p3p4 = torch.linalg.norm(vec_p3p4, dim=2, keepdim=True)
        len_p3p4 = torch.clamp(len_p3p4, min=1e-6)
        
        dotNormals = (vec_n1 * vec_n2).sum(dim=2, keepdim=True)
        cross_n1_n2 = torch.linalg.cross(vec_n1, vec_n2, dim=2)
        y = (cross_n1_n2 * vec_p3p4).sum(dim=2, keepdim=True) / len_p3p4
        
        theta = torch.atan2(y, dotNormals)

        diff = theta - self.theta
        diff = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi
        theta = self.theta + diff
        self.theta = theta
        
        delta_theta = theta - self.target_theta
        force_factor = -self.k_crease * self.origin_length * delta_theta * len_p3p4
        
        p1_forces_vectors = force_factor * (vec_n1 / lensq_n1)
        p2_forces_vectors = force_factor * (vec_n2 / lensq_n2)
        
        if self.use_projection:
            lensq_p3p4 = (vec_p3p4**2).sum(dim=2, keepdim=True)
            lensq_p3p4 = torch.clamp(lensq_p3p4, min=1e-6)

            u1 = (vec_p3p1 * vec_p3p4).sum(dim=2, keepdim=True) / lensq_p3p4
            u2 = (vec_p3p2 * vec_p3p4).sum(dim=2, keepdim=True) / lensq_p3p4

            p3_forces_vectors = -((1 - u1) * p1_forces_vectors + (1 - u2) * p2_forces_vectors)
            p4_forces_vectors = -(u1 * p1_forces_vectors + u2 * p2_forces_vectors)
        else:
            p3_forces_vectors = -(p1_forces_vectors + p2_forces_vectors)/2.0
            p4_forces_vectors = p3_forces_vectors

        self.total_forces.index_add_(1, self.face_idx0, p1_forces_vectors)
        self.total_forces.index_add_(1, self.face_idx1, p2_forces_vectors)
        self.total_forces.index_add_(1, self.face_idx2, p3_forces_vectors)
        self.total_forces.index_add_(1, self.face_idx3, p4_forces_vectors)

        spring_vectors = point2 - point1
        current_lengths = torch.linalg.norm(spring_vectors, dim=2, keepdim=True)
        current_lengths = torch.clamp(current_lengths, min=1e-3)
        displacements = current_lengths - self.rest_lengths
        force_magnitudes = - self.ea_rest_lengths * displacements
        force_vectors = force_magnitudes * (spring_vectors / current_lengths)

        self.total_forces.index_add_(1, self.line_idx1, force_vectors)  # spring force
        self.total_forces.index_add_(1, self.line_idx0, -force_vectors) # spring force

        # damping force
        point1velocity = self.velocities[:, self.line_idx0]
        point2velocity = self.velocities[:, self.line_idx1]
        damping_force =  self.damping_coef * (point2velocity - point1velocity)
        self.total_forces.index_add_(1, self.line_idx0, damping_force)
        self.total_forces.index_add_(1, self.line_idx1, -damping_force)
        
        accelerations = self.total_forces / self.masses
        self.velocities = self.velocities + accelerations * self.dt
        self.points = self.points + self.velocities * self.dt
        if self.total_forces.max() < self.min_force and abs(self.total_forces.min()) < self.min_force:
            return False
        return True

def origami_step_jax(state, params):
    points, velocities, current_theta = state
    lines, faces, rest_lengths, target_theta, origin_length_faces, masses, ea, k_crease, damping, dt = params
    
    total_forces = jnp.zeros_like(points)
    
    # --- Spring Forces (Axial) ---
    l_idx0 = lines[:, 0]
    l_idx1 = lines[:, 1]
    
    point1 = points[l_idx0]
    point2 = points[l_idx1]
    
    spring_vectors = point2 - point1
    current_lengths = jnp.linalg.norm(spring_vectors, axis=1, keepdims=True)
    current_lengths = jnp.maximum(current_lengths, 1e-3)
    
    displacements = current_lengths - rest_lengths
    force_magnitudes = -ea / rest_lengths * displacements
    spring_force_vectors = force_magnitudes * (spring_vectors / current_lengths)
    
    total_forces = total_forces.at[l_idx1].add(spring_force_vectors)
    total_forces = total_forces.at[l_idx0].add(-spring_force_vectors)
    
    # --- Damping Forces ---
    v1 = velocities[l_idx0]
    v2 = velocities[l_idx1]
    damping_force = 2.0 * damping * jnp.sqrt(ea / rest_lengths) * (v2 - v1)
    
    total_forces = total_forces.at[l_idx0].add(damping_force)
    total_forces = total_forces.at[l_idx1].add(-damping_force)
    
    # --- Crease Forces (Bending) ---
    f_idx0, f_idx1, f_idx2, f_idx3 = faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 3]
    
    p1 = points[f_idx0]
    p2 = points[f_idx1]
    p3 = points[f_idx2]
    p4 = points[f_idx3]
    
    vec_p3p1 = p1 - p3
    vec_p3p2 = p2 - p3
    vec_p3p4 = p4 - p3
    
    vec_n1 = jnp.cross(vec_p3p1, vec_p3p4, axis=1)
    vec_n2 = jnp.cross(vec_p3p4, vec_p3p2, axis=1)
    
    lensq_n1 = jnp.sum(vec_n1**2, axis=1, keepdims=True)
    lensq_n2 = jnp.sum(vec_n2**2, axis=1, keepdims=True)
    lensq_n1 = jnp.maximum(lensq_n1, 1e-6)
    lensq_n2 = jnp.maximum(lensq_n2, 1e-6)
    
    len_p3p4 = jnp.linalg.norm(vec_p3p4, axis=1, keepdims=True)
    len_p3p4 = jnp.maximum(len_p3p4, 1e-6)
    
    dotNormals = jnp.sum(vec_n1 * vec_n2, axis=1, keepdims=True)
    cross_n1_n2 = jnp.cross(vec_n1, vec_n2, axis=1)
    y = jnp.sum(cross_n1_n2 * vec_p3p4, axis=1, keepdims=True) / len_p3p4
    
    computed_theta = jnp.atan2(y, dotNormals)

    # Handle angle wrapping
    diff = computed_theta - current_theta
    diff = jnp.remainder(diff + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    new_theta = current_theta + diff
    delta_theta = new_theta - target_theta
    
    force_factor = -k_crease * origin_length_faces * delta_theta * len_p3p4
    
    p1_forces = force_factor * (vec_n1 / lensq_n1)
    p2_forces = force_factor * (vec_n2 / lensq_n2)
    lensq_p3p4 = jnp.sum(vec_p3p4**2, axis=1, keepdims=True)
    lensq_p3p4 = jnp.maximum(lensq_p3p4, 1e-6)

    u1 = jnp.sum(vec_p3p1 * vec_p3p4, axis=1, keepdims=True) / lensq_p3p4
    u2 = jnp.sum(vec_p3p2 * vec_p3p4, axis=1, keepdims=True) / lensq_p3p4

    p3_forces = -((1 - u1) * p1_forces + (1 - u2) * p2_forces)
    p4_forces = -(u1 * p1_forces + u2 * p2_forces)
    
    total_forces = total_forces.at[f_idx0].add(p1_forces)
    total_forces = total_forces.at[f_idx1].add(p2_forces)
    total_forces = total_forces.at[f_idx2].add(p3_forces)
    total_forces = total_forces.at[f_idx3].add(p4_forces)

    # --- Integration ---
    accelerations = total_forces / masses
    new_velocities = velocities + accelerations * dt
    new_points = points + new_velocities * dt
    
    return (new_points, new_velocities, new_theta), None

# 2. Simulation Loop (Single Instance)
# This sets up the time-stepping loop for one object.
def run_simulation_loop_jax(points, velocities, lines, faces, rest_lengths, 
                        current_theta, target_theta, origin_length_faces, 
                        mass, ea, k_crease, damping, dt, num_steps=3000):
    
    params = (lines, faces, rest_lengths, target_theta, origin_length_faces, 
              mass, ea, k_crease, damping, dt)
    
    initial_state = (points, velocities, current_theta)
    
    final_state, _ = jax.lax.scan(lambda s, _: origami_step_jax(s, params), initial_state, None, length=num_steps)
    
    return final_state[0] 

# 3. Batched Wrapper
# We use vmap here. in_axes maps inputs to dimensions. 
# 0 means "this argument has a batch dimension". 
# None means "this argument is shared across all batches".
@jax.jit(static_argnames="num_steps")
def run_simulation_batched(points, velocities, lines, faces, rest_lengths, 
                           current_theta, target_theta, origin_length_faces, 
                           mass, ea, k_crease, damping, dt, num_steps=5000):
    return jax.vmap(
        run_simulation_loop_jax, 
        in_axes=(0, 0, None, None, None, 0, 0, None, None, None, 0, None, None, None)
    )(points, velocities, lines, faces, rest_lengths, 
      current_theta, target_theta, origin_length_faces, 
      mass, ea, k_crease, damping, dt, num_steps)

class OrigamiObjectMatrixJax:
    def __init__(self, points: torch.Tensor,
                 lines: torch.Tensor,
                 faces: torch.Tensor,
                 target_theta: torch.Tensor,
                 mass: float = 1.0,
                 ea: float = 20.0,
                 k_crease: float = 0.7,
                 damping: float = 0.45,
                 fold_percent: float = 0.99,
                 dt: float = -1.0,
                 k_facet: float = 0.7):
        
        self.faces = faces.detach().clone()
        self.points_np = points.detach().clone().cpu().numpy()
        self.lines_np = lines.detach().clone().cpu().numpy()
        self.faces_np = faces.detach().clone().cpu().numpy()

        if self.points_np.ndim == 2:
            self.points_np = self.points_np[np.newaxis, ...]
            
        self.batch_size = self.points_np.shape[0]
        self.num_points = self.points_np.shape[1]
        
        target_np = (target_theta * fold_percent).detach().clone().cpu().numpy()
        if target_np.ndim == 2:
             target_np = np.tile(target_np[np.newaxis, ...], (self.batch_size, 1, 1))
            
        self.target_theta_np = target_np

        self.mass = mass
        self.ea = ea

        self.damping = damping
    
        self.points_jax = jnp.array(self.points_np)
        self.lines_jax = jnp.array(self.lines_np)
        self.faces_jax = jnp.array(self.faces_np)
        self.target_theta_jax = jnp.array(self.target_theta_np)
        self.k_crease_jax = jnp.full_like(self.target_theta_jax, k_crease)
        self.k_crease_jax = jnp.where(jnp.abs(self.target_theta_jax) <= 0.01, k_facet, self.k_crease_jax)

        self.velocities_jax = jnp.zeros_like(self.points_jax)
        self.theta_jax = jnp.zeros_like(self.target_theta_jax)
    
        p1 = self.points_jax[0, self.lines_np[:, 0]]
        p2 = self.points_jax[0, self.lines_np[:, 1]]
        self.rest_lengths = jnp.linalg.norm(p2 - p1, axis=1, keepdims=True)
        
        if dt < 0.0:
            min_len = float(jnp.min(self.rest_lengths))
            self.dt = 1.0 / (2.0 * np.pi * np.sqrt(self.ea / min_len))
        else:
            self.dt = dt

        p_f2 = self.points_jax[0, self.faces_np[:, 2]]
        p_f3 = self.points_jax[0, self.faces_np[:, 3]]
        self.origin_length_faces = jnp.linalg.norm(p_f2 - p_f3, axis=1, keepdims=True)

    def set_points(self, points: torch.Tensor):
        p_np = points.detach().clone().cpu().numpy()
        if p_np.ndim == 2:
            p_np = p_np[np.newaxis, ...]
        self.points_jax = jnp.array(p_np)
        self.velocities_jax = jnp.zeros_like(self.points_jax)

    def set_target_theta_from_np(self, target_theta: np.ndarray):
        t_np = target_theta
        if t_np.ndim == 2:
            t_np = np.tile(t_np[np.newaxis, ...], (self.batch_size, 1, 1))
        self.target_theta_jax = jnp.array(t_np)

    def run_steps(self, num_steps: int = 3000) -> torch.Tensor:
        final_points = run_simulation_batched(
            self.points_jax,
            self.velocities_jax,
            self.lines_jax,
            self.faces_jax,
            self.rest_lengths,
            self.theta_jax,
            self.target_theta_jax,
            self.origin_length_faces,
            self.mass,
            self.ea,
            self.k_crease_jax,
            self.damping,
            self.dt,
            num_steps
        )
        
        res = np.array(final_points)            
        return torch.from_numpy(res)

def get_3d_point(file_path):
    import json
    from visualization import visualize_simulation, visualize_point_cloud, ori_plotly_plot
    from util import pointcloud_sampling2, pointcloud_sampling_centroid, pointcloud_sampling3
    from data import get_data_extended

    points, edges, faces, target_theta, faces_indices_unique = get_data_extended(file_path)
    
    ori = OrigamiObjectMatrix(points.clone().unsqueeze(0), edges, faces, target_theta.unsqueeze(0))

    ori2 = OrigamiObjectMatrixJax(points.clone().unsqueeze(0), edges, faces, target_theta.unsqueeze(0))
    ori2points = ori2.run_steps(num_steps=10000)
    return ori2points

def get_3d_point_mesh(file_path):
    import json
    from visualization import visualize_simulation, visualize_point_cloud, ori_plotly_plot
    from util import pointcloud_sampling2, pointcloud_sampling_centroid, pointcloud_sampling3
    from data import get_data_extended

    points, edges, faces, target_theta, faces_indices_unique = get_data_extended(file_path)
    
    ori = OrigamiObjectMatrix(points.clone().unsqueeze(0), edges, faces, target_theta.unsqueeze(0))

    ori2 = OrigamiObjectMatrixJax(points.clone().unsqueeze(0), edges, faces, target_theta.unsqueeze(0))
    ori2points = ori2.run_steps(num_steps=10000)
    print(ori2.faces)
    print(faces_indices_unique)
    return ori2points, faces_indices_unique

if __name__ == "__main__":
    # points = torch.tensor([
    #     [-1., 0., 0.], [0.3, -2., 0.], [1., 0., 0.], [-0.5, 1.5, 0.],
    # ])

    # lines = torch.tensor([
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [1, 2],
    #     [2, 3],
    # ])

    # faces_indices = torch.tensor([
    #     [3, 1, 2, 0] # p1, p2, p3, p4, p3p4 is the crease
    # ])
    # target_theta = torch.tensor([
    #     [3.10], # ~pi for valley fold, -pi for mountain fold, 0 for no fold, and everything in between, corresponding to the crease p3p4
    # ])
    import json
    from visualization import visualize_simulation, visualize_point_cloud, ori_plotly_plot
    from util import pointcloud_sampling2, pointcloud_sampling_centroid, pointcloud_sampling3
    from data import get_data_extended

    points, edges, faces, target_theta, faces_indices_unique = get_data_extended("output_main4/output_sym_NONE/91_32.json")
    
    ori = OrigamiObjectMatrix(points.clone().unsqueeze(0), edges, faces, target_theta.unsqueeze(0))

    ori2 = OrigamiObjectMatrixJax(points.clone().unsqueeze(0), edges, faces, target_theta.unsqueeze(0))
    ori2points = ori2.run_steps(num_steps=3000)
    print(ori2points)
    
    visualize_simulation(ori, num_steps=3000, run_all_steps=True)
    print(ori.points)
    print("Max error: ", torch.abs(ori2points - ori.points).max())
    
    points_sampling = pointcloud_sampling2(ori2points, faces_indices_unique, num_subdivisions=2)

    visualize_point_cloud(points_sampling.squeeze())

    # points_sampling = pointcloud_sampling_centroid(ori2points, faces_indices_unique)

    # visualize_point_cloud(points_sampling.squeeze())

    # points_sampling = pointcloud_sampling3(ori2points, faces_indices_unique)

    # visualize_point_cloud(points_sampling.squeeze())