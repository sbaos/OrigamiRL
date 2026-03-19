import json
import os
from utils.save_map import save_to_json, make_output
from utils.get_map import get_map_value_from_file,load_from_json
from utils.rotate_map import rotate_graph
from ptu.ptu import Point, Edge
from tqdm import tqdm

MAP_HASH_FILE = "./map_hash.json"


def save_map_hash():
    with open(MAP_HASH_FILE, 'w') as f:
        json.dump(list(old_map_value), f)

def _compute_hash(active_edges):
    """Compute hash value from a list of (u_idx, v_idx, value) tuples."""
    sorted_edges = sorted(active_edges)
    value = 0
    for k, (u, v, val) in enumerate(sorted_edges):
        value += (k + 1) * (u * 100 + v) * val
    return round(value, 4)

def _get_active_edges_from_json(data):
    """Extract active edges from a JSON dict as (min_idx, max_idx, rounded_value) tuples."""
    active = []
    for edge in data["edges"]:
        val = edge["value"]
        if val == -999 or val == 0:
            continue
        u_idx = min(edge["u"], edge["v"])
        v_idx = max(edge["u"], edge["v"])
        active.append((u_idx, v_idx, round(val, 4)))
    return active

def get_map_value(points: list[Point], edges: list[Edge], boundary_nodes: list[Point], rows=None, cols=None):
    
    edges = [edges.clone() for edges in edges]
    if rows is None or cols is None:
        # Fallback: no rotation, compute hash from edges directly
        active = []
        for edge in edges:
            if edge.value == -999 or edge.value == 0:
                continue
            u_idx = min(edge.u.point_idx, edge.v.point_idx)
            v_idx = max(edge.u.point_idx, edge.v.point_idx)
            active.append((u_idx, v_idx, round(edge.value, 4)))
        return _compute_hash(active)
    
    # Convert live pattern to JSON dict
    data = make_output(points, edges, rows, cols, boundary_nodes)
    
    # Compute hash for original + all rotations (90°, 180°, 270°)
    hashes = [_compute_hash(_get_active_edges_from_json(data))]
    
    rotated = data
    for _ in range(3):  # 90°, 180°, 270°
        rotated = rotate_graph(rotated)
        hashes.append(_compute_hash(_get_active_edges_from_json(rotated)))
    
    return min(hashes)

def compare_map_previus_is_same(points, edges, rows, cols, boundary_nodes):
    value = get_map_value(points, edges, boundary_nodes, rows, cols)
    for v in old_map_value:
        if abs(v - value) < 100 or abs(v+value) < 100:
            return True
    return False

old_map_value = set()

def main(folder_path: str, output_folder: str):
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".json"):
            points, edges, rows, cols, boundary_nodes = load_from_json(os.path.join(folder_path, file))
            for i in range(len(edges)):
                if abs(edges[i].value) < 0.1:
                    edges[i].value = 0
            value = get_map_value(points, edges, boundary_nodes, rows, cols)
            if compare_map_previus_is_same(points, edges, rows, cols, boundary_nodes):
                continue
            save_to_json(points, edges, rows, cols, os.path.join(output_folder, file), boundary_nodes)
            old_map_value.add(value)
    save_map_hash()
    
if __name__ == "__main__":

    main("./output_new", "./output_new_filtered")
    print(len(os.listdir("./output_new")),len(os.listdir("./output_new_filtered")))
