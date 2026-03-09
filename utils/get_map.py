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