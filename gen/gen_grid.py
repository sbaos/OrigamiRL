from ptu.ptu import calc_ptu, Point, Edge
from ptu.ptu_2 import calc_ptu_2
from utils.get_map import load_from_json
from enum import Enum
import random
import numpy as np
from utils.save_map import save_to_json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SYMMETRY(Enum):
    NONE = "NONE"
    X = "X"
    Y = "Y"
    XY_LEFT = "XY\\"
    XY_RIGHT = "XY/"
    XXYY = "XXYY"
    
GRID_SIZE = 7
HAS_INIT_EDGE = True

__symmetric__ = None
PI = 3.13

def calc_sector_angles(root_point: Point, p1: Point, p2: Point, p3: Point) -> tuple[list[list[float]], list[list[float]], Point, Point, Point]:
    points = root_point.point_root + root_point.children + [p1, p2, p3]

    vecs = [p.position - root_point.position for p in points]
    angles = [np.arctan2(v[1], v[0]) for v in vecs]
    
    in_angles = root_point.in_diheral_angles + root_point.out_diheral_angles + [-999, -999, -999]

    sorted_pairs = sorted(zip(points, in_angles, angles), key=lambda x: x[2])
    sorted_points, sorted_in_angles, sorted_angles = zip(*sorted_pairs)
    n = len(sorted_points)

    p1_new = None
    p2_new = None
    p3_new = None
    i = 0
   
    max_iter = n * 3
    while p1_new is None or p2_new is None or p3_new is None:
        max_iter -= 1
        if max_iter <= 0:
            return ([], [], None, None, None)
        if sorted_points[i] in [p1,p2,p3]:
            if p3_new is not None:
                if sorted_points[i] in [p1,p2,p3]:
                    p1_new = sorted_points[i]
                    p2_new = [p for p in [p1,p2,p3] if p not in [p1_new, p3_new]][0]
            else:
                next_idx = (i + 1) % n
                if sorted_points[next_idx] not in [p1,p2,p3]:
                    p3_new = sorted_points[i]
        i = (i + 1) % n
        
    i1 = sorted_points.index(p1_new)
    i2 = sorted_points.index(p2_new)
    i3 = sorted_points.index(p3_new)

    def sector_angles_ccw(start_idx: int, end_idx: int, not_include: int):
        angles_seg = []
        in_seg = []
        idx = start_idx
        step = 1
        try:
            while True:
                if idx == not_include:
                    angles_seg = []
                    in_seg = []
                    idx = start_idx
                    step = -1
                    continue
                next_idx = (idx + step) % n
                u = sorted_points[idx].position - root_point.position
                v = sorted_points[next_idx].position - root_point.position
                angle_af = np.arctan2(v[1], v[0])
                angle_bf = np.arctan2(u[1], u[0])
                
                if angle_af < 0:
                    angle_af = angle_af + np.pi*2
                if angle_bf < 0:
                    angle_bf = angle_bf + np.pi*2

                angle = angle_af - angle_bf
                if angle < 0:
                    angle = angle + np.pi*2
                angles_seg.append(angle)
                if sorted_in_angles[idx] != -999:
                    in_seg.append(sorted_in_angles[idx])
                
                if next_idx == end_idx:
                    break
                idx = next_idx
        except:
            pass
        return angles_seg, in_seg
    
    seg1_d, seg1_in = sector_angles_ccw(i3, i1, i2)
    seg2_d, seg2_in = sector_angles_ccw(i1, i2, i3)
    seg3_d, seg3_in = sector_angles_ccw(i2, i3, i1)
    
    return ([seg1_d, seg2_d, seg3_d],
            [seg1_in, seg2_in, seg3_in],
            p1_new, p2_new, p3_new)

def norm_angles(angles: [float]):
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            if abs(math.pi - angles[i][j]) < 1e-2:
                angles[i][j] = PI
                continue
            if abs(math.pi/2 - angles[i][j]) < 1e-2:
                angles[i][j] = angles[i][j] - 1e-2
                continue
            if abs(0 - angles[i][j]) < 1e-2:
                angles[i][j] = 1e-2
                continue
            if abs(math.pi/2 - angles[i][j]) < 1e-2:
                angles[i][j] = angles[i][j] + 1e-2
                continue
    return angles 

def get_symmetry_position(point: Point, symmetry: SYMMETRY):
    if symmetry == SYMMETRY.X:
        return (point.position[0], -point.position[1], point.position[2])
    elif symmetry == SYMMETRY.Y:
        return (-point.position[0], point.position[1], point.position[2])
    elif symmetry == SYMMETRY.XY_LEFT:
        return (-point.position[1], -point.position[0], point.position[2])
    elif symmetry == SYMMETRY.XY_RIGHT:
        return (point.position[1], point.position[0], point.position[2])
    return (point.position[0], point.position[1], point.position[2])

def do_symmetry():
    pass

def is_on_symmetric_line(point: Point, symmetry: SYMMETRY):
    if symmetry == SYMMETRY.X:
        return point.position[1] == 0
    elif symmetry == SYMMETRY.Y:
        return point.position[0] == 0
    elif symmetry == SYMMETRY.XY_LEFT:
        return point.position[0] == -point.position[1]
    elif symmetry == SYMMETRY.XY_RIGHT:
        return point.position[0] == point.position[1]
    return False

def pick_points(boundary_nodes, level = 1 ):
    if level == 0:
        return random.choice(boundary_nodes)
    temp_boundary_nodes = [point for point in boundary_nodes if point.level == level]
    if len(temp_boundary_nodes) == 0:
        return None
    return random.choice(temp_boundary_nodes)

def make_direction_edge(u: Point, v: Point, edge: Edge):
    edge.u = u
    edge.v = v
    return edge    

def get_point_with_position(points: list[Point], position: tuple[float, float, float]):
    for point in points:
        if np.linalg.norm(point.position - position) < 1e-6:
            return point
    return None

def is_on_segment(point: Point, u: Point, v: Point) -> bool:
    """
    Check if point is on the segment uv (include u, v)
    """
    direction1 = v.position - u.position
    direction2 = point.position - u.position
    direction3 = point.position - v.position
    
    cross_result = np.cross(direction1, direction2)
    cross_magnitude = np.linalg.norm(cross_result) if isinstance(cross_result, np.ndarray) and cross_result.ndim > 0 else abs(cross_result)
    
    if cross_magnitude < 1e-6 and np.dot(direction2, direction3) <= 1e-6:
        return True
    return False

def is_linear(p1: Point, p2: Point, p3: Point):
    v1 = p2.position - p1.position
    v2 = p3.position - p1.position
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return abs(cross) < 1e-6

def is_two_points_coincide(point1: Point, point2: Point) -> bool:
    return np.linalg.norm(point1.position - point2.position) < 1e-6

def is_two_segment_connected(segment1: Edge, segment2: Edge) -> bool:
    if is_two_points_coincide(segment1.u, segment2.u) or\
        is_two_points_coincide(segment1.u, segment2.v) or\
        is_two_points_coincide(segment1.v, segment2.u) or\
        is_two_points_coincide(segment1.v, segment2.v):
        return True
    return False

def is_two_segment_overlap(segment1: Edge, segment2: Edge) -> bool:
    if is_on_segment(segment1.u,segment2.u,segment2.v) and is_on_segment(segment1.v,segment2.u,segment2.v):
        return True
    elif is_on_segment(segment2.u,segment1.u,segment1.v) and is_on_segment(segment2.v,segment1.u,segment1.v):
        return True
    return False

def is_two_segment_intersec(segment1: Edge, segment2: Edge) -> bool:
    if is_two_segment_tangent(segment1,segment2):
        return True
    
    u = segment1.u.position - segment1.v.position
    u_2d = np.array([u[1], -u[0]]) 
    c1 = -u_2d.dot(segment1.u.position[:2])
    
    v = segment2.u.position - segment2.v.position
    v_2d = np.array([v[1], -v[0]]) 
    c2 = -v_2d.dot(segment2.u.position[:2])

    A = np.array([u_2d, v_2d])
    
    if abs(np.linalg.det(A)) < 1e-9:
        return False
    
    b = np.array([c1, c2])
    x = np.linalg.solve(A, b)

    if is_on_segment(Point(x[0], x[1]), segment1.u, segment1.v) and is_on_segment(Point(x[0], x[1]), segment2.u, segment2.v):
        return True
    return False

def is_uv_of_segment(u: Point, v: Point, edge: Edge)->bool:
    if (is_two_points_coincide(u, edge.u) and is_two_points_coincide(v, edge.v)) or \
        (is_two_points_coincide(u, edge.v) and is_two_points_coincide(v, edge.u)):
        return True
    return False    

def is_two_segment_tangent(segment1: Edge, segment2: Edge) -> bool:
    if is_on_segment(segment1.u,segment2.u,segment2.v):
        return True
    elif is_on_segment(segment1.v,segment2.u,segment2.v):
        return True
    elif is_on_segment(segment2.u,segment1.u,segment1.v):
        return True
    elif is_on_segment(segment2.v,segment1.u,segment1.v):
        return True
    return False

def get_intersect_point(segment1: Edge, segment2: Edge, points):
    if is_two_segment_tangent(segment1,segment2):
        if is_on_segment(segment1.u,segment2.u,segment2.v):
            return segment1.u
        elif is_on_segment(segment1.v,segment2.u,segment2.v):
            return segment1.v
        elif is_on_segment(segment2.u,segment1.u,segment1.v):
            return segment2.u
        elif is_on_segment(segment2.v,segment1.u,segment1.v):
            return segment2.v
    
    u = segment1.u.position - segment1.v.position
    u_2d = np.array([u[1], -u[0]])  # perpendicular vector
    c1 = -u_2d.dot(segment1.u.position[:2])
    
    v = segment2.u.position - segment2.v.position
    v_2d = np.array([v[1], -v[0]])  # perpendicular vector
    c2 = -v_2d.dot(segment2.u.position[:2])

    A = np.array([u_2d, v_2d])
    
    # Check if lines are parallel
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    
    b = np.array([c1, c2])
    x = np.linalg.solve(A, b)

    if is_on_segment(Point(x[0], x[1]), segment1.u, segment1.v) and is_on_segment(Point(x[0], x[1]), segment2.u, segment2.v):
        for point in points:
            if np.linalg.norm(point.position[:2] - np.array([x[0], x[1]])) < 1e-6:
                return point
        return None
    return None

def get_sub_points_of_edge(edge: Edge, points: [Point]) -> [Point]:
    sub_points = []
    for point in points:
        if is_on_segment(point, edge.u, edge.v):
            sub_points.append(point)
    sub_points.sort(key=lambda point: np.linalg.norm(point.position - edge.u.position))
    sub_points.remove(edge.u)
    return sub_points

def  make_edge_longer_fixed_length(root_point: Point, other_point: Point, theta: float, points: [Point], edges: [Edge], num_step: int) -> tuple[[Edge], Point]:
    u = other_point.position - root_point.position
    edges_clone = [edge.clone() for edge in edges]
    new_point_position = root_point + u*num_step
    new_point = other_point.clone()
    new_point.position = new_point_position
    for point in points:
        if np.linalg.norm(point.position - new_point_position) < 1e-6:
            if HAS_INIT_EDGE:
                pass
            return edges_clone,point
    return edges_clone,None
        
def make_edge_longer_as_possible(root_point: Point, other_point: Point, theta: float, points: [Point], edges: [Edge]) -> tuple[[Edge], Point]:
    u = other_point.position - root_point.position
    edges_clone = [edge.clone() for edge in edges]
    for i in range(0,GRID_SIZE*2):
        for edge in edges_clone:
            if edge.value <= -9:
                continue
            temp_point = other_point.clone()
            temp_point.position = temp_point.position + u*i

#kiem tra xem co che duoc len doan chua co khong
            if HAS_INIT_EDGE:
                pass        
            if is_two_segment_tangent(edge,Edge(root_point,temp_point,0)) and\
                not is_two_segment_overlap(edge,Edge(root_point,temp_point,0)) and\
                not is_on_segment(root_point,edge.u,edge.v):
                intersect_point = get_intersect_point(edge,Edge(root_point,temp_point,0), points)
                if np.linalg.norm(intersect_point.position - temp_point.position) > 1e-6:
                    continue
                sub_points = get_sub_points_of_edge(Edge(other_point,temp_point,0), points)  
                if len(sub_points) == 0:
                    return edges_clone,intersect_point
                
                start_point = other_point
                idx = 0
                for idx in range(len(sub_points)):
                    end_point = sub_points[idx]
                    start_point.add_children(end_point,theta)
                    end_point.add_parent(start_point,theta)
                    for edge_ in edges_clone:
                        if is_uv_of_segment(start_point, end_point, edge_):
                            edge_.value = theta
                            start_point = end_point
                            
                return edges_clone,intersect_point
            elif is_two_segment_intersec(edge,Edge(root_point,temp_point,0)) and\
                    not is_two_segment_tangent(edge,Edge(root_point,temp_point,0)):
                return edges_clone,None
    raise Exception("No intersect found")       
    
def expand_points_3(root_point: Point, points: [Point], edges: [Edge], boundary_nodes: [Point])->tuple[Point,Point,Point]:
    def get_possible_points():
        possible_points = []
        for edge in edges:
            if edge.u != root_point and edge.v != root_point:
                continue
            if edge.u in root_point.point_root or edge.v in root_point.point_root:
                continue
            if edge.value != -999:
                if edge.u in boundary_nodes and edge.u != root_point:
                    possible_points.append(edge.u)
                if edge.v in boundary_nodes and edge.v != root_point:
                    possible_points.append(edge.v)
                continue
            possible_points.append(edge.u if edge.u != root_point else edge.v)
        return possible_points

    possible_points = get_possible_points()
    if len(possible_points) < 3:
        return None, None, None
    p1, p2, p3 = random.sample(possible_points, 3)
    if __symmetric__ is not None and is_on_symmetric_line(root_point,__symmetric__):
        p1 = random.sample(possible_points, 1)[0]
        p2 = get_point_with_position(possible_points,get_symmetry_position(p1,__symmetric__))
        num_try = 100
        while num_try > 0 and (is_on_symmetric_line(p1,__symmetric__) or p2 is None):
            p1 = random.sample(possible_points, 1)[0]
            num_try -= 1
            p2 = get_point_with_position(possible_points,get_symmetry_position(p1,__symmetric__))

            if p2 is None or p2 == p1:
                continue

        num_try = 100
        p3 = random.sample(possible_points, 1)[0]
        while num_try > 0 and not is_on_symmetric_line(p3,__symmetric__):
            p3 = random.sample(possible_points, 1)[0]
            num_try -= 1
    return p1, p2, p3

def expand_point_2(root_point: Point, points: [Point], edges: [Edge])->tuple[Point,Point]:
    pass
    def get_possible_points():
        possible_points = []
        for edge in edges:
            if edge.u != root_point and edge.v != root_point:
                continue
            if edge.u in root_point.point_root or edge.v in root_point.point_root:
                continue
            possible_points.append(edge.u if edge.u != root_point else edge.v)
        return possible_points
    
    possible_points = get_possible_points()
    p1, p2 = random.sample(possible_points, 2)
    return p1, p2

def handle_merge(root_point: Point,other_point: Point,value: float, points: [Point], temp_edges: [Edge]):
    real_root_point = root_point
    for edge in temp_edges:
        if edge.u == other_point and is_on_segment(edge.v, root_point, other_point):
            real_root_point = edge.v
        elif edge.v == other_point and is_on_segment(edge.u, root_point, other_point):
            real_root_point = edge.u
        
    if real_root_point.is_actived:
            other_point.add_parent(root_point,value)
    
def update_edges(u: Point,v: Point,theta, edges: [Edge]):
    updated = False
    for edge in edges:
        if edge.u == u and edge.v == v:
            edge.value = theta
            updated = True
        elif edge.v == u and edge.u == v:
            edge.value = theta
            updated = True
    if not updated:
        edges.append(Edge(u,v,theta))

def show_map(points, edges, rows, cols, boundary_nodes):
    fig, ax = plt.subplots(figsize=(7, 7))
    boundary_idxs = set(p.point_idx for p in boundary_nodes)

    for edge in edges:
        x = [edge.u.position[0], edge.v.position[0]]
        y = [edge.u.position[1], edge.v.position[1]]
        if edge.value == -999:
            ax.plot(x, y, color='#dddddd', linewidth=0.5, linestyle='--')
        elif edge.value == 0:
            ax.plot(x, y, color='black', linewidth=2)
        else:
            c = 'blue' if edge.value > 0 else 'red'
            alpha = min(abs(edge.value) / math.pi, 1.0)
            ax.plot(x, y, color=c, alpha=alpha, linewidth=2)
            mx, my = (x[0]+x[1])/2, (y[0]+y[1])/2
            ax.text(mx, my, f'{edge.value:.1f}', fontsize=6, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

    for p in points:
        color = '#333333' if p.is_border else '#cc44cc' if p.point_idx in boundary_idxs else '#4a90e2'
        ax.plot(p.position[0], p.position[1], 'o', color=color, markersize=8)
        ax.text(p.position[0], p.position[1], str(p.point_idx), fontsize=5, ha='center', va='center', color='white')

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Gen Grid Viewer')
    plt.tight_layout()
    plt.show()

def is_valid_result(root_point: Point,result: list[float]):
    temp_point = root_point.clone()
    for i in range(len(result)):
        temp_point.add_children(temp_point.clone(),result[i])
    degree = sum([1 if angle > 0.1 else 0 for angle in result])
    value = [res if res > 0.1 else 0 for res in result][0]
    if degree == 1 and len(root_point.point_root) == 1 and abs(root_point.in_diheral_angles[0]-value) < 0.1:
        return True
    elif degree == 1 and len(root_point.point_root) == 1:
        return False
    total_degree = degree + len(root_point.point_root)
    if total_degree < 3:
        return False
    positive = sum([1 if angle > 0 else 0 for angle in result+root_point.in_diheral_angles])
    negative = sum([1 if angle < 0 else 0 for angle in result+root_point.in_diheral_angles])
    degree = count_degree(temp_point)
    
    if degree % 2 == 0 and abs(positive-negative) != 2:
        return False
    return True

def count_degree(point: Point):
    degree =sum([1 if abs(angle) > 0.1 else 0 for angle in  point.in_diheral_angles+point.out_diheral_angles])
    return degree

def get_min_level(points: [Point]):
    min_level = 99999999
    for point in points:
        min_level = min(min_level, point.level)
    return min_level

def gen_pattern(file_path: str, symmetry: SYMMETRY, N = 30, extend_full = True, edge_extend_as_posible = True):
    
    global __symmetric__
    __symmetric__ = symmetry
    points, edges, rows, cols, boundary_nodes = load_from_json(file_path)

    num_try = 0
    
    while N > 0 and ((extend_full and len(boundary_nodes) > 0) or (not extend_full)) :
        temp_edges = [edge.clone() for edge in edges]
        N -= 1
        expand_point = pick_points(boundary_nodes, get_min_level(boundary_nodes))

        num_try += 1

        if expand_point is None or num_try % 10 == 0:
            continue
        expand_point.is_actived = True
        p1_0, p2_0, p3_0 = expand_points_3(expand_point, points, temp_edges, boundary_nodes)

        p1,p2,p3 = p1_0, p2_0, p3_0
        if p1 is None or p2 is None or p3 is None:
            expand_point.is_actived= False
            continue
        sector_angles = []
        list_in_diheral_angles = []
        sector_angles, list_in_diheral_angles, p1, p2, p3 = calc_sector_angles(expand_point, p1, p2, p3)
        sector_angles = norm_angles(sector_angles)
        # print("expand_point: ",expand_point.point_idx," p1: ",p1.point_idx," p2: ",p2.point_idx," p3: ",p3.point_idx)
        list_in_diheral_angles = norm_angles(list_in_diheral_angles)
        _, M1, M2 = calc_ptu_2(sector_angles, list_in_diheral_angles)
        if len(M1) == 0 or len(M2) == 0:
            expand_point.is_actived = False
            continue
        
        valid_result = []
        if is_valid_result(expand_point,M1):
            valid_result.append(M1)
        if is_valid_result(expand_point,M2):
            valid_result.append(M2)
        if len(valid_result) == 0:
            expand_point.is_actived = False
            continue

        chose_out_diheral_angle = random.choice(valid_result)

        update_edges(expand_point,p1,chose_out_diheral_angle[0],temp_edges)
        update_edges(expand_point,p2,chose_out_diheral_angle[1],temp_edges)
        update_edges(expand_point,p3,chose_out_diheral_angle[2],temp_edges)

        if edge_extend_as_posible:
            temp_edges, p1 = make_edge_longer_as_possible(expand_point, p1, chose_out_diheral_angle[0], points, temp_edges)
            temp_edges, p2 = make_edge_longer_as_possible(expand_point, p2, chose_out_diheral_angle[1], points, temp_edges)
            temp_edges, p3 = make_edge_longer_as_possible(expand_point, p3, chose_out_diheral_angle[2], points, temp_edges)
        if p1.is_actived and p1 not in boundary_nodes:
            expand_point.is_actived = False
            continue
        if p2.is_actived and p2 not in boundary_nodes:
            expand_point.is_actived = False
            continue
        if p3.is_actived and p3 not in boundary_nodes:
            expand_point.is_actived = False
            continue
        boundary_nodes.remove(expand_point)
        
        handle_merge(expand_point,p1,chose_out_diheral_angle[0],points, temp_edges)
        handle_merge(expand_point,p2,chose_out_diheral_angle[1],points, temp_edges)
        handle_merge(expand_point,p3,chose_out_diheral_angle[2],points, temp_edges)

        if not p1.is_border and p1 not in boundary_nodes:
            boundary_nodes.append(p1)
            p1.is_actived = True
        if not p2.is_border and p2 not in boundary_nodes:
            boundary_nodes.append(p2)
            p2.is_actived = True
        if not p3.is_border and p3 not in boundary_nodes:
            boundary_nodes.append(p3)
            p3.is_actived = True

        p1.level = expand_point.level + 2
        p2.level = expand_point.level + 2
        p3.level = expand_point.level + 2

        p1.is_actived = True
        p2.is_actived = True
        p3.is_actived = True

        p1_0.add_parent(expand_point,chose_out_diheral_angle[0])
        p2_0.add_parent(expand_point,chose_out_diheral_angle[1])
        p3_0.add_parent(expand_point,chose_out_diheral_angle[2])

        expand_point.add_children(p1_0, chose_out_diheral_angle[0])
        expand_point.add_children(p2_0, chose_out_diheral_angle[1])
        expand_point.add_children(p3_0, chose_out_diheral_angle[2])

        edges = temp_edges
    
    for point in boundary_nodes:
        theta = -999
        for edge in edges:
            if is_uv_of_segment(point,point.point_root[0],edge):
                theta = edge.value
                break
        edges, p = make_edge_longer_as_possible(point.point_root[0], point, theta, points, edges)
    for point in points:
        if point.is_border:
            continue
        degree = count_degree(point)
        if degree == 3 or degree == 1:
            return [],[],7,7,[]
        elif degree == 2:
            connected = []
            angles = []
            for idx, parent in enumerate(point.point_root):
                if abs(point.in_diheral_angles[idx]) > 0.1:
                    connected.append(parent)
                    angles.append(point.in_diheral_angles[idx])
            for idx, child in enumerate(point.children):
                if abs(point.out_diheral_angles[idx]) > 0.1:
                    connected.append(child)
                    angles.append(point.out_diheral_angles[idx])
            if len(connected) == 2:
                if not is_linear(connected[0], point, connected[1]) or abs(angles[0] - angles[1]) >= 0.1:
                    return [], [], 7, 7, []
        elif degree % 2 == 0 and degree>0:
            positive = sum([1 if angle > 0.1 else 0 for angle in point.in_diheral_angles+point.out_diheral_angles])
            negative = sum([1 if angle < -0.1 else 0 for angle in point.in_diheral_angles+point.out_diheral_angles])
            if abs(positive-negative) != 2:
                return [], [], 7, 7, []
    # save_to_json(points, edges, rows, cols, "./output/basic_3.json", boundary_nodes)
    return points, edges, rows, cols, boundary_nodes
    

# goc = 0 thi bo luon