import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptu.ptu import Point, Edge

def get_edge_value_by_u_v(edges, u, v):
    for edge in edges:
        if edge["u"] == u and edge["v"] == v:
            return edge["value"]
        elif edge["u"] == v and edge["v"] == u:
            return edge["value"]
    return None

def add_all_connection(points, node_connections, edges):
    for point in points:
        point_connections = node_connections.get(str(point.point_idx), {})
        point.point_root = [points[i] for i in point_connections.get("in_edges", [])]
        point.in_diheral_angles = [get_edge_value_by_u_v(edges, point.point_idx, i) for i in point_connections.get("in_edges", [])]
        point.children = [points[i] for i in point_connections.get("out_edges", [])]

def load_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    rows = metadata.get("rows", 0)
    cols = metadata.get("cols", 0)
    boundary_nodes_idx = data.get("boundary_nodes", [])
    border_points_idx = data.get("border_points", [])
    node_connections = data.get("node_connections", {})
    nodes_data = data.get("nodes", {})
    points = []
    for i in range(len(nodes_data)):
        coords = nodes_data[str(i)]
        is_border = i in border_points_idx
        points.append(Point(coords[0], coords[1], 0, point_idx=i, is_border=is_border))

    edges_data = data.get("edges", [])
    edges = []
    for edge_data in edges_data:
        edge = Edge(
            u=points[edge_data["u"]],
            v=points[edge_data["v"]],
            value=edge_data["value"],
            attributes=edge_data["attributes"],
            edge_type=edge_data["edge_type"]
        )
        edges.append(edge)
    add_all_connection(points, node_connections, edges_data)
    boundary_nodes = [points[i] for i in boundary_nodes_idx]
    return points, edges, rows, cols, boundary_nodes

def _compute_hash(active_edges):
    """Compute hash value from a list of (u_idx, v_idx, value) tuples."""
    sorted_edges = sorted(active_edges)
    value = 0
    for k, (u, v, val) in enumerate(sorted_edges):
        value += (k + 1) * (u * 100 + v) * val
    return round(value, 4)

def _rotate_idx_180(idx, rows, cols):
    r, c = idx // cols, idx % cols
    return (rows - 1 - r) * cols + (cols - 1 - c)

def _rotate_idx_90cw(idx, n):
    r, c = idx // n, idx % n
    return c * n + (n - 1 - r)

def _rotate_idx_270cw(idx, n):
    r, c = idx // n, idx % n
    return (n - 1 - c) * n + r

def _rotate_edges(active, rotate_fn, *args):
    """Apply a rotation function to all edges and return new edge tuples."""
    rotated = []
    for u, v, val in active:
        new_u = rotate_fn(u, *args)
        new_v = rotate_fn(v, *args)
        rotated.append((min(new_u, new_v), max(new_u, new_v), val))
    return rotated

def get_map_value_from_file(file_path: str):
    points, edges, rows, cols, boundary_nodes = load_from_json(file_path)
    active = []
    for edge in edges:
        if edge.value == -999 or edge.value == 0:
            continue
        u_idx = min(edge.u.point_idx, edge.v.point_idx)
        v_idx = max(edge.u.point_idx, edge.v.point_idx)
        active.append((u_idx, v_idx, round(edge.value, 4)))
    
    # Compute hash for all rotations, return the minimum (canonical form)
    hashes = [_compute_hash(active)]
    
    # 180° rotation (works for any grid)
    rotated_180 = _rotate_edges(active, _rotate_idx_180, rows, cols)
    hashes.append(_compute_hash(rotated_180))
    
    # 90° and 270° rotations (only for square grids)
    if rows == cols:
        n = rows
        rotated_90 = _rotate_edges(active, _rotate_idx_90cw, n)
        hashes.append(_compute_hash(rotated_90))
        rotated_270 = _rotate_edges(active, _rotate_idx_270cw, n)
        hashes.append(_compute_hash(rotated_270))
    
    return min(hashes)

def init_map(rows, cols):
    points = [Point(x - cols/2, y - rows/2, 0) for y in range(rows) for x in range(cols)]
    edges = []
    return points, edges, rows, cols


if __name__ == "__main__":
    points, edges, rows, cols = load_from_json("./pattern/may_bay_don_gian.json")
    print(points)
    print(edges)
    print(rows)
    print(cols)