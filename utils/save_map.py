import os
import json


def make_output(points, edges, rows, cols, boundary_nodes=None, node_connections=None):
    """
    Create output dict from points and edges.
    
    Args:
        points: List of Point objects
        edges: List of Edge objects
        rows: Grid rows
        cols: Grid cols
        boundary_nodes: List of boundary node indices (optional, defaults to [])
        node_connections: Dict of node connections (optional, defaults to {})
    """
    def make_node(point):
        return [float(point.position[0]), float(point.position[1])]
    
    def make_edge(edge):
        # Handle both Point objects and int indices for u, v
        u_idx = edge.u.point_idx if hasattr(edge.u, 'point_idx') else edge.u
        v_idx = edge.v.point_idx if hasattr(edge.v, 'point_idx') else edge.v
        
        # Determine direction
        if edge.value == 0:
            direction = "border"
        elif edge.value == -999 or edge.value is None:
            direction = "none"
        else:
            direction = f"{u_idx} -> {v_idx}"
        
        return {
            "u": u_idx,
            "v": v_idx,
            "value": float(edge.value) if edge.value is not None and edge.value != -999 else 0,
            "attributes": edge.attributes if hasattr(edge, 'attributes') and edge.attributes else [],
            "edge_type": edge.edge_type if hasattr(edge, 'edge_type') else "",
            "direction": direction
        }
    
    output = {
        "metadata": {
            "rows": rows,
            "cols": cols
        },
        "nodes": {
            str(i): make_node(point) for i, point in enumerate(points)
        },
        "edges": [make_edge(edge) for edge in edges],
        # Convert boundary_nodes to indices if they are Point objects
        "boundary_nodes": [
            (bn.point_idx if hasattr(bn, 'point_idx') else bn) 
            for bn in (boundary_nodes if boundary_nodes is not None else [])
        ],
        "node_connections": node_connections if node_connections is not None else {}
    }
    
    return output


def save_to_json(points, edges, rows, cols, filepath, boundary_nodes=None, node_connections=None):
    """
    Save points and edges to a JSON file.
    
    Args:
        points: List of Point objects
        edges: List of Edge objects
        rows: Grid rows
        cols: Grid cols
        filepath: Output file path
        boundary_nodes: List of boundary node indices (optional, defaults to [])
        node_connections: Dict of node connections (optional, defaults to {})
    """
    output = make_output(points, edges, rows, cols, boundary_nodes, node_connections)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    
    print(f"Saved to {filepath}")
    return filepath