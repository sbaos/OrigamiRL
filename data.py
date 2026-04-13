import json
from rotate_graph import rotate_graph
import torch

def get_data(json_file_path="test_fold/box_boat.json"):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    points, lines, faces, target_theta = [], [], [], []
    for key, node in data["nodes"].items():
        points.append([node[0], node[1], 0.0])

    data["edges"] = sorted(data["edges"], key=lambda e: (e["u"], e["v"]))
    for line in data["edges"]:
        lines.append([line["u"], line["v"]])
        p0, p1, p2, p3 = line["attributes"][0], line["attributes"][1], line["attributes"][2], line["attributes"][3]
        if p0 is not None and p1 is not None and p2 is not None and p3 is not None:
            faces.append([p0, p1, p2, p3])
            target_theta.append([line["value"]])

    points = torch.tensor(points)
    lines = torch.tensor(lines)
    faces_indices = torch.tensor(faces)
    target_theta_gt = torch.tensor(target_theta)
    return points, lines, faces_indices, target_theta_gt

def _extract_data(data):
    points, lines, faces, target_theta = [], [], [], []
    data["nodes"] = dict(sorted(data["nodes"].items(), key=lambda item: int(item[0])))
    for key, node in data["nodes"].items():
        points.append([float(node[0]), float(node[1]), 0.0])

    data["edges"] = sorted(data["edges"], key=lambda e: (e["u"], e["v"]))
    faces_unique = set()
    for line in data["edges"]:
        lines.append([line["u"], line["v"]])
        p0, p1, p2, p3 = line["attributes"][0], line["attributes"][1], line["attributes"][2], line["attributes"][3]
        if p0 is not None and p1 is not None and p2 is not None and p3 is not None:
            faces.append([p0, p1, p2, p3])
            target_theta.append([line["value"]])
            faces_unique.add(tuple(sorted([p0, p2, p3])))
            faces_unique.add(tuple(sorted([p1, p2, p3])))

    points = torch.tensor(points)
    lines = torch.tensor(lines)
    faces_indices = torch.tensor(faces)
    target_theta_gt = torch.tensor(target_theta)
    faces_indices_unique = torch.tensor(list(faces_unique))
    return points, lines, faces_indices, target_theta_gt, faces_indices_unique

def get_data_extended(json_file_path="test_fold/box_boat.json"):
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return _extract_data(data)

def get_data_extended_rotations(json_file_path="test_fold/box_boat.json"):
    with open(json_file_path, "r") as f:
        data = json.load(f)
    x0 = _extract_data(data)
    data = rotate_graph(data)
    x1 = _extract_data(data)
    data = rotate_graph(data)
    x2 = _extract_data(data)
    data = rotate_graph(data)
    x3 = _extract_data(data)
    return x0, x1, x2, x3
    

def create_sphere_pointcloud(center, radius, num_points=1000):
    """
    Creates a point cloud distributed on a sphere surface using Fibonacci lattice.
    
    Args:
        center: (x, y, z) tuple or tensor
        radius: float
        num_points: int, controls density (number of points)
        
    Returns:
        torch.Tensor: (num_points, 3) tensor of points
    """
    indices = torch.arange(0, num_points, dtype=torch.float32) + 0.5
    
    phi = torch.acos(1 - 2*indices/num_points)
    theta = torch.pi * (1 + 5**0.5) * indices
    
    x = radius * torch.cos(theta) * torch.sin(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(phi)
    
    points = torch.stack([x, y, z], dim=1)
    
    # Ensure center is a tensor
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=torch.float32)
    
    points = points + center
    
    return points

if __name__ == "__main__":
    
    from visualization import visualize_point_cloud
    center_point = (0, 0, 0)
    sphere_radius = 2.0
    density_param = 100
    
    sphere_points = create_sphere_pointcloud(center_point, sphere_radius, density_param)
    print(sphere_points.shape)
    
    visualize_point_cloud(sphere_points, show=True)

    points, lines, faces, target_theta_gt, faces_indices_unique = get_data_extended()
    print(points.shape, lines.shape, faces.shape, target_theta_gt.shape, faces_indices_unique.shape)