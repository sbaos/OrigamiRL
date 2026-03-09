from gen.gen_grid import gen_pattern
from gen.gen_grid import SYMMETRY
import random
from utils.save_map import save_to_json
from ptu.ptu import Point, Edge

random.seed(2)

num_i = 15
num_j = 1

file_path = "./pattern/XY_LEFT_{num}.json"
symmetry = SYMMETRY.Y

old_map_value = []
ignore_value = []
def get_map_value(points: list[Point], edges: list[Edge], boundary_nodes: list[Point]):
    active = []
    for edge in edges:
        if edge.value == -999 or edge.value == 0:
            continue
        u_idx = min(edge.u.point_idx, edge.v.point_idx)
        v_idx = max(edge.u.point_idx, edge.v.point_idx)
        active.append((u_idx, v_idx, round(edge.value, 4)))
    active.sort()
    value = 0
    for k, (u, v, val) in enumerate(active):
        value += (k + 1) * (u * 100 + v) * val
    return round(value, 4)

def compare_map_previus_is_same(points, edges, rows, cols, boundary_nodes):
    value = get_map_value(points,edges,boundary_nodes)
    for v in old_map_value:
        if abs(v-value) < 1e-2:
            return True
    return False

i = 14
j = 0

while i < num_i:
    file_path = "./pattern/Y_"+str(i)+".json"
    j = 0
    num_try = 100
    while j < num_j and num_try > 0 :
        print("Try: ",i,j)
        output_path = "./output_5/Y_"+str(i)+"_"+str(j)+".json"
        num_try -=1
        try:
            points, edges, rows, cols, boundary_nodes = gen_pattern(file_path, symmetry)
            if len(points) == 0 or points is None:
                continue
            value = get_map_value(points,edges,boundary_nodes)
            if not compare_map_previus_is_same(points, edges, rows, cols, boundary_nodes):
                save_to_json(points, edges, rows, cols, output_path, boundary_nodes)
                old_map_value.append(get_map_value(points,edges,boundary_nodes))
            else:
                # ignore_value.append(value)
                j -=1
            j+=1
        except:
            import traceback
            traceback.print_exc()
            continue
    i+=1

# print(old_map_value)
# print(ignore_value)