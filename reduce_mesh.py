import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
import sys

def calculate_face_colors(verts, faces):
    """Calculate face colors based on normals for a more premium look."""
    # Ensure arrays are float64 for normal calculation
    verts = verts.astype(np.float64)
    
    # Vectors for two edges of each triangle
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # Normals = cross product
    normals = np.cross(v1 - v0, v2 - v0)
    
    # Normalize
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm!=0)
    
    # Map normals [-1, 1] to [0, 255] for RGB
    # We use a slight shift and scale to get a nice shade
    colors = (normals + 1) / 2 * 255
    return [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in colors]

def get_mesh_trace(verts, faces, name):
    """Create a Plotly Mesh3d trace."""
    face_colors = calculate_face_colors(verts, faces)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        facecolor=face_colors,
        opacity=1.0,
        flatshading=True,
        name=name,
        showlegend=True
    )

def main():
    parser = argparse.ArgumentParser(description="Reduce mesh triangle count and visualize.")
    parser.add_argument("input", help="Path to the input .off file")
    parser.add_argument("--target", type=int, default=150, help="Target triangle count (default: 150)")
    parser.add_argument("--output", help="Optional path to save the reduced .off file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        sys.exit(1)

    print(f"Loading {args.input}...")
    # Load original mesh
    original_mesh = o3d.io.read_triangle_mesh(args.input)
    if not original_mesh.has_triangles():
        print("Error: Could not load triangles from file. Check if it's a valid .off file.")
        sys.exit(1)
        
    orig_verts = np.asarray(original_mesh.vertices)
    orig_faces = np.asarray(original_mesh.triangles)
    print(f"Original mesh: {len(orig_verts)} vertices, {len(orig_faces)} triangles.")

    # Simplify mesh
    print(f"Reducing mesh to approximately {args.target} triangles...")
    reduced_mesh = original_mesh.simplify_quadric_decimation(target_number_of_triangles=args.target)
    
    # Cleanup: Remove unreferenced vertices, degenerate triangles, etc.
    print("Post-processing: Removing unreferenced vertices and degenerate triangles...")
    reduced_mesh.remove_unreferenced_vertices()
    reduced_mesh.remove_degenerate_triangles()
    reduced_mesh.remove_duplicated_vertices()
    reduced_mesh.remove_duplicated_triangles()
    
    red_verts = np.asarray(reduced_mesh.vertices)
    red_faces = np.asarray(reduced_mesh.triangles)
    print(f"Reduced mesh: {len(red_verts)} vertices, {len(red_faces)} triangles.")

    # Save if requested
    if args.output:
        print(f"Saving reduced mesh to {args.output}...")
        o3d.io.write_triangle_mesh(args.output, reduced_mesh)

    # Visualization
    print("Preparing visualization...")
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]],
        subplot_titles=(f"Original ({len(orig_faces)} triangles)", f"Reduced ({len(red_faces)} triangles)")
    )

    # Add original mesh to first subplot
    fig.add_trace(get_mesh_trace(orig_verts, orig_faces, "Original"), row=1, col=1)
    
    # Add reduced mesh to second subplot
    fig.add_trace(get_mesh_trace(red_verts, red_faces, "Reduced"), row=1, col=2)

    # Unified scene styling for a premium look
    scene_config = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='rgb(20, 20, 25)' # Dark background
    )

    fig.update_layout(
        title_text=f"Mesh Decimation Comparison: {os.path.basename(args.input)}",
        title_x=0.5,
        template="plotly_dark",
        scene=scene_config,
        scene2=scene_config,
        margin=dict(l=0, r=0, b=0, t=60),
        height=600
    )

    print("Showing visualization. Close the window to exit.")
    fig.show()

if __name__ == "__main__":
    main()
