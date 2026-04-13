import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

# --- Helper Functions ---

def _to_numpy(tensor_or_array):
    if hasattr(tensor_or_array, 'cpu'):
        return tensor_or_array.cpu().detach().numpy()
    return tensor_or_array

def _get_topology(faces):
    """
    Given faces (N, 4), return the triangulation indices for front and back meshes.
    """
    faces = _to_numpy(faces)
    p0 = faces[:, 0]
    p1 = faces[:, 1]
    p2 = faces[:, 2]
    p3 = faces[:, 3]

    # Front (Blue) - Counter-Clockwise
    # Triangles: (p2, p0, p3) and (p2, p3, p1) based on original logic
    i_front = np.concatenate([p2, p2])
    j_front = np.concatenate([p0, p3])
    k_front = np.concatenate([p3, p1])

    # Back (Gray) - Clockwise
    i_back = np.concatenate([p2, p2])
    j_back = np.concatenate([p3, p1])
    k_back = np.concatenate([p0, p3])
    
    return i_front, j_front, k_front, i_back, j_back, k_back

def _compute_offset_back_points(points, i_front, j_front, k_front, epsilon=0.005):
    """
    Compute points for the back mesh by offsetting along vertex normals.
    Normals are computed from the front mesh topology.
    """
    # Compute Face Normals
    v0 = points[i_front]
    v1 = points[j_front]
    v2 = points[k_front]
    
    vec1 = v1 - v0
    vec2 = v2 - v0
    # Cross product for face normals
    face_normals = np.cross(vec1, vec2)
    
    # Accumulate vertex normals
    vertex_normals = np.zeros_like(points)
    np.add.at(vertex_normals, i_front, face_normals)
    np.add.at(vertex_normals, j_front, face_normals)
    np.add.at(vertex_normals, k_front, face_normals)
    
    # Normalize
    vn_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (vn_norms + 1e-9)
    
    points_back = points - epsilon * vertex_normals
    return points_back

def _get_wireframe_coords(points, i_indices, j_indices, k_indices):
    """
    Extract unique edges for wireframe visualization.
    """
    # Create array of edges: (N_tris * 3, 2)
    e1 = np.stack([i_indices, j_indices], axis=1)
    e2 = np.stack([j_indices, k_indices], axis=1)
    e3 = np.stack([k_indices, i_indices], axis=1)
    
    all_edges = np.concatenate([e1, e2, e3], axis=0)
    # Sort vertex indices within each edge to handle (u,v) vs (v,u)
    all_edges.sort(axis=1)
    # Filter unique edges
    unique_edges = np.unique(all_edges, axis=0)
    
    # Vectorized construction of coordinate lists
    pts_start = points[unique_edges[:, 0]]
    pts_end = points[unique_edges[:, 1]]
    
    M = len(unique_edges)
    # We need to insert None between segments. 
    # Structure: [x1, x2, None, x1, x2, None, ...]
    
    xe = np.empty((M, 3), dtype=object)
    xe[:, 0] = pts_start[:, 0]
    xe[:, 1] = pts_end[:, 0]
    xe[:, 2] = None
    
    ye = np.empty((M, 3), dtype=object)
    ye[:, 0] = pts_start[:, 1]
    ye[:, 1] = pts_end[:, 1]
    ye[:, 2] = None

    ze = np.empty((M, 3), dtype=object)
    ze[:, 0] = pts_start[:, 2]
    ze[:, 1] = pts_end[:, 2]
    ze[:, 2] = None
    
    return xe.flatten(), ye.flatten(), ze.flatten()

def _create_origami_traces(points, topology):
    """
    Create the 4 Plotly traces (Front Mesh, Back Mesh, Front Edges, Back Edges).
    """
    i_front, j_front, k_front, i_back, j_back, k_back = topology
    points_back = _compute_offset_back_points(points, i_front, j_front, k_front)
    
    # 1. Front Mesh
    trace_front = go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        i=i_front, j=j_front, k=k_front,
        color='rgb(65, 145, 255)', 
        opacity=1.0, name='Front', flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.9, roughness=0.1, specular=0.3)
    )
    
    # 2. Back Mesh
    trace_back = go.Mesh3d(
        x=points_back[:, 0], y=points_back[:, 1], z=points_back[:, 2],
        i=i_back, j=j_back, k=k_back,
        color='rgb(120, 120, 120)',
        opacity=1.0, name='Back', flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.9, roughness=0.1, specular=0.3)
    )
    
    # 3. Front Edges
    xe_f, ye_f, ze_f = _get_wireframe_coords(points, i_front, j_front, k_front)
    trace_edges_f = go.Scatter3d(
        x=xe_f, y=ye_f, z=ze_f,
        mode='lines', line=dict(color='black', width=4),
        name='Front Edges', hoverinfo='skip'
    )
    
    # 4. Back Edges
    # Note: Using front topology indices for edges is correct as connectivity is same.
    xe_b, ye_b, ze_b = _get_wireframe_coords(points_back, i_front, j_front, k_front)
    trace_edges_b = go.Scatter3d(
        x=xe_b, y=ye_b, z=ze_b,
        mode='lines', line=dict(color='black', width=4),
        name='Back Edges', hoverinfo='skip'
    )
    
    return [trace_front, trace_back, trace_edges_f, trace_edges_b]


# --- Main Functions ---

def visualize_simulation(ori, num_steps=10000, run_all_steps=False, save_path=None, show=True):
    """
    Visualize simulation using Plotly Animation
    Args:
        ori: OrigamiObjectMatrix
        run_all_steps: If True, run for full num_steps steps even if converged.
    """
    
    # --- Pre-calculate Topology ---
    topology = _get_topology(ori.faces)

    # --- Run Simulation and Collect Frames ---
    frames_raw = []
    
    # Store initial state
    points_initial = _to_numpy(ori.points[0])
    frames_raw.append(points_initial.copy())

    print(f"Simulating up to {num_steps} steps...")
    
    for _ in range(num_steps):
        continuing = ori.step()
        
        points_cur = _to_numpy(ori.points[0])
        frames_raw.append(points_cur.copy())
        
        if not continuing and not run_all_steps:
            print("Simulation converged.")
            break
        
    total_raw_frames = len(frames_raw)
    print(f"Total simulated steps: {total_raw_frames}")

    # --- Subsample Frames ---
    TARGET_FRAMES = 200 
    
    if total_raw_frames > TARGET_FRAMES:
        indices = np.linspace(0, total_raw_frames - 1, TARGET_FRAMES, dtype=int)
        frames_data = [frames_raw[i] for i in indices]
    else:
        indices = np.arange(total_raw_frames)
        frames_data = frames_raw
        
    print(f"Visualizing {len(frames_data)} frames (subsampled).")

    # --- Calculate Cubic Global Bounding Box ---
    all_points = np.concatenate(frames_data, axis=0) 
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    
    center = (min_coords + max_coords) / 2.0
    ranges = max_coords - min_coords
    max_range = np.max(ranges)
    
    # Add padding (e.g., 5%)
    half_size = (max_range * 1.05) / 2.0
    
    fixed_x_range = [center[0] - half_size, center[0] + half_size]
    fixed_y_range = [center[1] - half_size, center[1] + half_size]
    fixed_z_range = [center[2] - half_size, center[2] + half_size]

    # --- Build Plotly Figure ---
    
    # Create Initial Data
    data = _create_origami_traces(frames_data[0], topology)
    
    # Create Animation Frames
    frames = []
    for k, pts in enumerate(frames_data):
        # We need to recreate the traces for each frame
        frame_traces = _create_origami_traces(pts, topology)
        frames.append(go.Frame(data=frame_traces, name=str(k)))

    # Layout
    fig = go.Figure(data=data, frames=frames)
    
    fig.update_layout(
        title="Origami Simulation",
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(visible=False, range=fixed_x_range), 
            yaxis=dict(visible=False, range=fixed_y_range),
            zaxis=dict(visible=False, range=fixed_z_range),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.4))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor='white',
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": True}, 
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Step:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                    "label": str(indices[k]),
                    "method": "animate"
                }
                for k in range(len(frames))
            ]
        }]
    )

    if show:
        fig.show()
    if save_path:
        fig.write_html(save_path)

def ori_plotly_plot(points, faces, save_path=None, show=True, pointsclound2=None):
    """
    Static visualization of an origami state.
    """
    points = _to_numpy(points)
    faces = _to_numpy(faces)
    
    topology = _get_topology(faces)
    traces = _create_origami_traces(points, topology)

    if pointsclound2 is not None:
        pointsclound2 = _to_numpy(pointsclound2)
        
        trace = go.Scatter3d(
            x=pointsclound2[:, 0], y=pointsclound2[:, 1], z=pointsclound2[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=points[:, 2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=1
            )
        )
        traces.append(trace)
    
    fig = go.Figure(data=traces)

    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.4))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor='white'
    )

    if show:
        fig.show()
    if save_path:
        fig.write_html(save_path)

def visualize_point_cloud(points, save_path=None, show=True):
    """
    Visualization of a point cloud.
    """
    points = _to_numpy(points)
    
    trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=points[:, 2],                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=1
        )
    )
    
    fig = go.Figure(data=[trace])

    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.4))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor='white'
    )

    if show:
        fig.show()
    if save_path:
        fig.write_html(save_path)


def visualize_range_dataset(dataset, faces, start_idx=0, end_idx=None, save_path=None, show=True):
    """
    Visualize a range of dataset items using a Plotly slider.
    
    Args:
        dataset: A PhysicEngineDataset (or similar) where dataset[i][0] returns points.
        faces: Face topology tensor/array (N, 4).
        start_idx: Start index (inclusive).
        end_idx: End index (exclusive). Defaults to len(dataset).
        save_path: Optional path to save the figure as HTML.
        show: Whether to show the figure in a browser.
    """
    if end_idx is None:
        end_idx = len(dataset)
    
    start_idx = max(0, start_idx)
    end_idx = min(len(dataset), end_idx)
    
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx} >= end_idx={end_idx}")

    faces = _to_numpy(faces)
    topology = _get_topology(faces)

    # Collect all points in the range
    frames_data = []
    label_data = []
    for i in range(start_idx, end_idx):
        pts = _to_numpy(dataset[i][0])
        frames_data.append(pts)
        label_data.append(f"{dataset[i][2]}, Num rotations: {dataset[i][3]}, Percentage: {dataset[i][4]}")

    num_frames = len(frames_data)
    print(f"Visualizing dataset indices [{start_idx}, {end_idx}) — {num_frames} items.")

    # --- Calculate Cubic Global Bounding Box ---
    all_points = np.concatenate(frames_data, axis=0)
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)

    center = (min_coords + max_coords) / 2.0
    ranges = max_coords - min_coords
    max_range = np.max(ranges)

    half_size = (max_range * 1.05) / 2.0

    fixed_x_range = [center[0] - half_size, center[0] + half_size]
    fixed_y_range = [center[1] - half_size, center[1] + half_size]
    fixed_z_range = [center[2] - half_size, center[2] + half_size]

    # --- Build Plotly Figure ---
    data = _create_origami_traces(frames_data[0], topology)

    frames = []
    for k, pts in enumerate(frames_data):
        frame_traces = _create_origami_traces(pts, topology)
        frames.append(go.Frame(data=frame_traces, name=str(k)))

    fig = go.Figure(data=data, frames=frames)

    fig.update_layout(
        title=f"Dataset Range [{start_idx}, {end_idx})",
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(visible=False, range=fixed_x_range),
            yaxis=dict(visible=False, range=fixed_y_range),
            zaxis=dict(visible=False, range=fixed_z_range),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.4))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        paper_bgcolor='white',
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Index:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                    "label": str(start_idx + k) + label_data[start_idx+k],
                    "method": "animate"
                }
                for k in range(num_frames)
            ]
        }]
    )

    if show:
        fig.show()
    if save_path:
        fig.write_html(save_path)


def save_origami_png(points, faces, save_prefix="origami", show=False):
    """
    Saves the origami shape as PNG files from different view angles.
    Args:
        points: Points tensor/array (N, 3).
        faces: Face topology tensor/array (F, 4).
        save_prefix: Prefix for the saved PNG files.
        show: Whether to show the figure in a browser (at the last perspective).
    """
    points = _to_numpy(points)
    faces = _to_numpy(faces)
    
    topology = _get_topology(faces)
    traces = _create_origami_traces(points, topology)
    
    fig = go.Figure(data=traces)
    
    # Define common view angles
    views = {
        "perspective": dict(x=1.5, y=1.5, z=1.5),
        "top": dict(x=0, y=0, z=2.5),
        "front": dict(x=2.5, y=0, z=0),
        "side": dict(x=0, y=2.5, z=0),
        "iso": dict(x=1.5, y=-1.5, z=1.5),
    }
    
    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor='white'
    )

    # Create directory if it doesn't exist
    dir_name = os.path.dirname(save_prefix)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    for name, eye in views.items():
        fig.update_layout(scene_camera=dict(eye=eye))
        output_path = f"{save_prefix}{name}.png"
        # Using scale for higher resolution
        fig.write_image(output_path, scale=2)
        print(f"Saved view {name} to {output_path}")

    if show:
        fig.update_layout(scene_camera=dict(eye=views["perspective"]))
        fig.show()


if __name__ == "__main__":
    from data import get_data
    points, lines, faces, target_theta_gt = get_data()
    ori_plotly_plot(points, faces, show=True, pointsclound2=points)
