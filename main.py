from gen.gen_grid import gen_pattern
from gen.gen_grid import SYMMETRY
import random
import json
import os
import glob
import csv
import torch
from utils.save_map import save_to_json, make_output
from utils.get_map import get_map_value_from_file
from utils.rotate_map import rotate_graph
from ptu.ptu import Point, Edge
from tqdm import tqdm

random.seed(2)

NUM_PATTERN = 16
NUM_SAMPLE_OF_PATTERN = 40

START_PATTERN = 5
END_PATTERN = 15

START_SAMPLE = 1
NUM_TRY = 26
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
        if abs(v - value) < 300:
            return True
    return False

def main1(pattern_folder: str, output_folder_name: str, symmetry = SYMMETRY.Y, edge_extend_as_posible = True):
    pbar_i = tqdm(range(START_PATTERN,END_PATTERN), desc="Patterns", position=0)
    for i in pbar_i:
        file_path = pattern_folder+"/"+str(i)+".json"
        j = START_SAMPLE
        num_try_ = NUM_TRY
        while j < NUM_SAMPLE_OF_PATTERN and num_try_ > 0 :
            num_try_ -=1
            try:
                pbar_k = tqdm(range(1,MAX_NUM_STEP), desc=f"  Pattern {i} | j={j} try={num_try_}", position=1, leave=False)
                for k in pbar_k:
                    pbar_k.set_description(f"  Pattern {i} | j={j} try={num_try_}")
                    output_path = output_folder_name+"/"+str(i)+"_"+str(j)+".json"
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


def main2(pattern_folder: str, output_folder_name_ = "./output_foldable"):
    pbar_sym = tqdm([SYMMETRY.Y], desc=f"Symmetry", position=0)
    pbar_i = tqdm(range(START_PATTERN,END_PATTERN), desc="Patterns", position=1)
    for symmetry in pbar_sym:
        output_folder_name = output_folder_name_+"/output_sym_"+symmetry.name
        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name, exist_ok=True)
        for i in pbar_i:
            file_path = pattern_folder+"/"+str(i)+".json"
            j = START_SAMPLE
            num_try_ = NUM_TRY
            while j < NUM_SAMPLE_OF_PATTERN and num_try_ > 0 :
                num_try_ -=1
                try:
                    pbar_k = tqdm(range(1,MAX_NUM_STEP), desc=f"  Pattern {i} | j={j} try={num_try_}", position=2, leave=False)
                    for k in pbar_k:
                        pbar_k.set_description(f"  Pattern {i} | j={j} try={num_try_}")
                        output_path = output_folder_name+"/"+str(i)+"_"+str(j)+".json"
                        points, edges, rows, cols, boundary_nodes = gen_pattern(file_path, 
                                                                                symmetry=symmetry,
                                                                                N=k,
                                                                                extend_full=k<50,
                                                                                edge_extend_as_posible=random.random()<0.5
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


def main3(file_path = "./pattern/sample.json", output_folder_name =  "/output_sample_1", symmetry = SYMMETRY.Y, edge_extend_as_posible = True):
    try:
        N = 1
        while N<100:
            N+=1
            points, edges, rows, cols, boundary_nodes = gen_pattern(file_path, 
                                                                    symmetry=symmetry,
                                                                    N=3,
                                                                    extend_full=True,
                                                                    edge_extend_as_posible=edge_extend_as_posible
                                                                    )
            if len(points) == 0 or points is None:
                print("Failed to generate pattern from ", file_path)
                continue
            save_to_json(points, edges, rows, cols, output_folder_name, boundary_nodes)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
def get_all_map_value_in_folder(folder_name: str):
    old_map_value = set()
    if os.path.exists(folder_name):
        for file in os.listdir(folder_name):
            if file.endswith(".json"):
                file_path = os.path.join(folder_name, file)
                value = get_map_value_from_file(file_path)
                old_map_value.add(value)
    return old_map_value

def get_points_from_file(file_path):
    points = []
    try:
        if file_path.endswith('.obj'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            points.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif file_path.endswith('.off'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    first_line = lines[0].strip()
                    if first_line.startswith('OFF'):
                        if len(first_line) > 3:
                            # Fix for OFF1512 84 0 format missing newline
                            lines[0] = first_line[3:] + '\n'
                            header_line = 0
                        else:
                            header_line = 1
                    else:
                        header_line = 0
                        
                    while header_line < len(lines) and (lines[header_line].startswith('#') or not lines[header_line].strip()):
                        header_line += 1
                        
                    if header_line < len(lines):
                        parts = lines[header_line].strip().split()
                        if parts:
                            try:
                                num_vertices = int(parts[0])
                                max_idx = min(header_line + 1 + num_vertices, len(lines))
                                for i in range(header_line + 1, max_idx):
                                    parts = lines[i].strip().split()
                                    if len(parts) >= 3:
                                        points.append((float(parts[0]), float(parts[1]), float(parts[2])))
                            except ValueError as ve:
                                pass
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    # Sample down to 1000 points to make chamfer computation faster
    if len(points) > 1000:
        points = random.sample(points, 1000)
        
    return torch.tensor(points, dtype=torch.float32)


def main4(pattern_folder: str, target_folder: str = "./target", output_folder_name_: str = "./output_main4", csv_output: str = "./result.csv"):
    from solver import get_3d_point
    from loss import invariant_chamfer_loss_2
    pbar_sym = tqdm([SYMMETRY.Y,SYMMETRY.XY_LEFT,SYMMETRY.NONE], desc=f"Symmetry", position=0)
    pbar_i = tqdm(range(START_PATTERN, END_PATTERN), desc="Patterns", position=1)
    
    # Collect target files: 5 from train per category (up to 40 folders), with FolderName
    target_files = []  # list of (file_path, folder_name)
    if os.path.exists(target_folder):
        categories = sorted([d for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))])[:40]
        for cat in categories:
            cat_path = os.path.join(target_folder, cat)
            train_files = glob.glob(os.path.join(cat_path, "train", "*.off")) + glob.glob(os.path.join(cat_path, "train", "*.obj"))
            
            # Select 5 from train per category
            cat_files = train_files[:5]
            
            if not cat_files:  # If no train subfolder, grab files from category folder directly
                cat_files = (glob.glob(os.path.join(cat_path, "*.off")) + glob.glob(os.path.join(cat_path, "*.obj")))[:5]
            
            for f_path in cat_files:
                target_files.append((f_path, cat))
            
    print(f"Selected {len(target_files)} background targets from {len(categories)} categories for comparison.")
    
    # Pre-parse points to avoid doing it per pattern: (file_path, folder_name, points_tensor)
    targets_data = [(obj, folder_name, get_points_from_file(obj)) for obj, folder_name in target_files]
    
    headers = ['pattern_path']
    for i in range(1, 6):
        headers.extend([f'obj{i}_name', f'chamfer_ob_{i}', f'obj{i}_path', f'obj{i}_folder'])
        
    with open(csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    for symmetry in pbar_sym:
        output_folder_name = output_folder_name_+"/output_sym_"+symmetry.name
        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name, exist_ok=True)
            
        for i in pbar_i:
            file_path = pattern_folder+"/"+str(i)+".json"
            if not os.path.exists(file_path):
                continue
            j = START_SAMPLE
            num_try_ = NUM_TRY
            while j < NUM_SAMPLE_OF_PATTERN and num_try_ > 0:
                num_try_ -= 1
                try:
                    pbar_k = tqdm(range(1, MAX_NUM_STEP), desc=f"  Pattern {i} | j={j} try={num_try_}", position=2, leave=False)
                    for k in pbar_k:
                        pbar_k.set_description(f"  Pattern {i} | j={j} try={num_try_}")
                        output_path = output_folder_name+"/"+str(i)+"_"+str(j)+".json"
                        points, edges, rows, cols, boundary_nodes = gen_pattern(file_path, 
                                                                                symmetry=symmetry,
                                                                                N=k,
                                                                                extend_full=k<50,
                                                                                edge_extend_as_posible=random.random()<0.5
                                                                                )
                        if len(points) == 0 or points is None:
                            continue
                        value = get_map_value(points, edges, boundary_nodes, rows, cols)
                        if not compare_map_previus_is_same(points, edges, rows, cols, boundary_nodes):
                            j+=1
                            save_to_json(points, edges, rows, cols, output_path, boundary_nodes)
                            old_map_value.add(value)
                            if len(old_map_value) % 50 == 0:
                                save_map_hash()
                            
                            pattern_3d_points = get_3d_point(output_path)
                            
                            distances = []
                            for obj_path, folder_name, target_pts in targets_data:
                                if len(target_pts) == 0 or len(pattern_3d_points) == 0:
                                    distances.append((obj_path, folder_name, float('inf')))
                                    continue
                                try:
                                    loss = invariant_chamfer_loss_2(pattern_3d_points, target_pts, return_mean=True)
                                    distances.append((obj_path, folder_name, loss.item()))
                                except Exception as e:
                                    distances.append((obj_path, folder_name, float('inf')))
                                    
                            valid_distances = [(path, fname, loss) for path, fname, loss in distances if loss != float('inf')]
                            valid_distances.sort(key=lambda x: x[2])
                            # Deduplicate by FolderName: keep only the best (lowest chamfer) per folder
                            seen_folders = set()
                            top_5 = []
                            for path, fname, loss in valid_distances:
                                if fname not in seen_folders:
                                    seen_folders.add(fname)
                                    top_5.append((path, fname, loss))
                                if len(top_5) >= 5:
                                    break
                            
                            row = [output_path]
                            for rank in range(5):
                                if rank < len(top_5):
                                    obj_p, fname, l = top_5[rank]
                                    row.extend([os.path.basename(obj_p), str(l), os.path.relpath(obj_p, target_folder), fname])
                                else:
                                    row.extend(['', '', '', ''])
                                    
                            with open(csv_output, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(row)                                
                    pbar_k.close()
                except Exception as e:
                    # print(e)
                    # raise e
                    continue

if __name__ == "__main__":
    old_map_value = get_all_map_value_in_folder("") #./output_new_filtered")
    # old_map_value = get_all_map_value_in_folder("./output_8") | old_map_value
    # old_map_value = get_all_map_value_in_folder("./output_9") | old_map_value
    # old_map_value = get_all_map_value_in_folder("./output_90") | old_map_value
    # old_map_value = get_all_map_value_in_folder("./output_91") | old_map_value
    # old_map_value = get_all_map_value_in_folder("./output_92") | old_map_value
    print(f"Loaded {len(old_map_value)} unique map values from existing patterns.")
    # main3()
    main4("pattern_9x9", "./target", "./output_9x9", "./result_9x9.csv")
    # main2(
        # pattern_folder="./pattern")
    
    # main1(
    #     pattern_folder="./pattern_merge",
    #     output_folder_name="./output/output_sym_Y",
    #     symmetry=SYMMETRY.Y,
    #     edge_extend_as_posible=False
    # )

    # main1(
    #     pattern_folder="./pattern_merge",
    #     output_folder_name="./output/output_sym_X",
    #     symmetry=SYMMETRY.X,
    #     edge_extend_as_posible=False
    # )

    # main1(
    #     pattern_folder="./pattern_merge",
    #     output_folder_name="./output/output_sym_YX",
    #     symmetry=SYMMETRY.XY_LEFT,
    #     edge_extend_as_posible=False
    # )

    # main1(
    #     pattern_folder="./pattern_merge",
    #     output_folder_name="./output/output_sym_XY",
    #     symmetry=SYMMETRY.XY_RIGHT,
    #     edge_extend_as_posible=False
    # )

    # main1(
    #     pattern_folder="./pattern_merge",
    #     output_folder_name="./output/output_sym_NONE",
    #     symmetry=SYMMETRY.NONE,
    #     edge_extend_as_posible=False
    # )
# obj,
# 3point for obj in obj...:
# chàmer = 
# csv : 
# path | label1 | label2 | label3 | label