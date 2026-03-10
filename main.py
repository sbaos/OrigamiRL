from gen.gen_grid import gen_pattern
from gen.gen_grid import SYMMETRY
import random
import json
import os
from utils.save_map import save_to_json
from utils.get_map import get_map_value_from_file
from ptu.ptu import Point, Edge
from tqdm import tqdm

random.seed(2)

NUM_PATTERN = 70
NUM_SAMPLE_OF_PATTERN = 30

START_PATTERN = 0
END_PATTERN = 70

START_SAMPLE = 0
NUM_TRY = 20
MAX_NUM_STEP = 100
file_path = "./pattern/XY_LEFT_{num}.json"
symmetry = SYMMETRY.Y

MAP_HASH_FILE = "./map_hash.json"

def load_map_hash():
    if os.path.exists(MAP_HASH_FILE):
        with open(MAP_HASH_FILE, 'r') as f:
            return set(json.load(f))
    else:
        with open(MAP_HASH_FILE, 'w') as f:
            json.dump([], f)
        return set()

def save_map_hash():
    with open(MAP_HASH_FILE, 'w') as f:
        json.dump(list(old_map_value), f)

old_map_value = load_map_hash()
ignore_value = []
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

def _get_active_edges(edges):
    """Extract active edges as (min_idx, max_idx, rounded_value) tuples."""
    active = []
    for edge in edges:
        if edge.value == -999 or edge.value == 0:
            continue
        u_idx = min(edge.u.point_idx, edge.v.point_idx)
        v_idx = max(edge.u.point_idx, edge.v.point_idx)
        active.append((u_idx, v_idx, round(edge.value, 4)))
    return active

def _rotate_edges(active, rotate_fn, *args):
    """Apply a rotation function to all edges and return new edge tuples."""
    rotated = []
    for u, v, val in active:
        new_u = rotate_fn(u, *args)
        new_v = rotate_fn(v, *args)
        rotated.append((min(new_u, new_v), max(new_u, new_v), val))
    return rotated

def get_map_value(points: list[Point], edges: list[Edge], boundary_nodes: list[Point], rows=None, cols=None):
    active = _get_active_edges(edges)
    
    if rows is None or cols is None:
        return _compute_hash(active)
    
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

def compare_map_previus_is_same(points, edges, rows, cols, boundary_nodes):
    value = get_map_value(points, edges, boundary_nodes, rows, cols)
    for v in old_map_value:
        if abs(v - value) < 100:
            return True
    return False

def main1(pattern_folder: str, output_folder_name: str, symmetry = SYMMETRY.Y, edge_extend_as_posible = True):
    pbar_i = tqdm(range(START_PATTERN,END_PATTERN), desc="Patterns", position=0)
    for i in pbar_i:
        file_path = pattern_folder+"/Y_"+str(i)+".json"
        j = START_SAMPLE
        num_try_ = NUM_TRY
        while j < NUM_SAMPLE_OF_PATTERN and num_try_ > 0 :
            num_try_ -=1
            try:
                pbar_k = tqdm(range(1,MAX_NUM_STEP), desc=f"  Pattern {i} | j={j} try={num_try_}", position=1, leave=False)
                for k in pbar_k:
                    pbar_k.set_description(f"  Pattern {i} | j={j} try={num_try_}")
                    output_path = output_folder_name+"/Y_"+str(i)+"_"+str(j)+".json"
                    points, edges, rows, cols, boundary_nodes = gen_pattern(file_path, 
                                                                            symmetry=symmetry,
                                                                            N=k,
                                                                            extend_full=k<50,
                                                                            edge_extend_as_posible=edge_extend_as_posible
                                                                            )
                    if len(points) == 0 or points is None:
                        continue
                    value = get_map_value(points, edges, boundary_nodes, rows, cols)
                    if not compare_map_previus_is_same(points, edges, rows, cols, boundary_nodes):
                        save_to_json(points, edges, rows, cols, output_path, boundary_nodes)
                        old_map_value.add(value)
                        if len(old_map_value) % 50 == 0:
                            save_map_hash()
                        j+=1
                pbar_k.close()
            except:
                # import traceback
                # traceback.print_exc()
                continue

def get_all_map_value_in_folder(folder_name: str):
    old_map_value = set()
    if os.path.exists(folder_name):
        for file in os.listdir(folder_name):
            if file.endswith(".json"):
                file_path = os.path.join(folder_name, file)
                value = get_map_value_from_file(file_path)
                old_map_value.add(value)
    return old_map_value

if __name__ == "__main__":
    old_map_value = get_all_map_value_in_folder("./output_7")
    old_map_value = get_all_map_value_in_folder("./output_8") | old_map_value

    main1(
        pattern_folder="./pattern",
        output_folder_name="./output_9",
        symmetry=SYMMETRY.Y,
        edge_extend_as_posible=False
    )