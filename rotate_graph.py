"""
Rotate a crease pattern graph by 90 degrees counter-clockwise.

Usage:
    python rotate_graph.py <input.json> <output.json>
    python rotate_graph.py <input_dir/> <output_dir/>   # bulk mode (recursive)

The rotation maps (x, y) -> (-y, x) for node coordinates.
Node IDs are remapped to maintain row-major ordering (top-left to bottom-right).
Edge types are swapped: Horizontal <-> Vertical, Diagonal\\ <-> Diagonal/.
"""

from copy import deepcopy
import json
import argparse
from pathlib import Path


def build_node_id_map(rows, cols):
    """
    Build a mapping from old node ID to new node ID after 90° CCW rotation.

    In a rows x cols grid, node at grid position (r, c) has ID = r * cols + c.
    A 90° CCW rotation maps grid (r, c) -> (cols - 1 - c, r).
    After rotation, the grid becomes cols x rows, so the new ID is:
        new_id = (cols - 1 - c) * rows + r
    """
    old_to_new = {}
    for r in range(rows):
        for c in range(cols):
            old_id = r * cols + c
            new_r = cols - 1 - c
            new_c = r
            new_id = new_r * rows + new_c  # new grid is cols x rows
            old_to_new[old_id] = new_id
    return old_to_new


def rotate_edge_type(edge_type):
    """Swap edge types for a 90° rotation."""
    mapping = {
        "Horizontal": "Vertical",
        "Vertical": "Horizontal",
        "Diagonal\\": "Diagonal/",
        "Diagonal/": "Diagonal\\",
    }
    return mapping.get(edge_type, edge_type)


def rotate_node_coords(x, y):
    """Rotate a point 90° CCW: (x, y) -> (-y, x)."""
    return [-y, x]


def rotate_graph(data):
    """Rotate the entire graph 90° counter-clockwise."""
    result = deepcopy(data)
    rows = data["metadata"]["rows"]
    cols = data["metadata"]["cols"]

    # Build the node ID remapping
    old_to_new = build_node_id_map(rows, cols)

    theta_val = {}
    for i in range(len(data["edges"])):
        u = data["edges"][i]["u"]
        v = data["edges"][i]["v"]

        theta_val[(min(u, v), max(u, v))] = data["edges"][i]["value"]

    for i in range(len(result["edges"])):
        u = result["edges"][i]["u"]
        v = result["edges"][i]["v"]
        u = old_to_new[u]
        v = old_to_new[v]
        result["edges"][i]["value"] = theta_val[(min(u, v), max(u, v))]

    return result


def rotate_file(src, dst, rotations):
    """Load, rotate, and save a single JSON file."""
    with open(src, "r") as f:
        data = json.load(f)
    for _ in range(rotations):
        data = rotate_graph(data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Rotate a crease pattern graph by 90° counter-clockwise. "
                    "If both input and output are directories, all JSON files are "
                    "rotated recursively, preserving the directory structure."
    )
    parser.add_argument("input", help="Path to input JSON file or directory")
    parser.add_argument("output", help="Path to output JSON file or directory")
    parser.add_argument(
        "-n", "--times", type=int, default=1,
        help="Number of 90° rotations to apply (default: 1)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rotations = args.times % 4  # 4 rotations = identity

    if input_path.is_dir():
        # Bulk mode: glob all JSON files recursively
        json_files = list(input_path.rglob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_path}")
            return
        output_path.mkdir(parents=True, exist_ok=True)
        for src in json_files:
            dst = output_path / src.relative_to(input_path)
            rotate_file(src, dst, rotations)
            print(f"  {src} -> {dst}")
        print(f"\nDone: rotated {len(json_files)} file(s) {args.times}x90°.")
    else:
        # Single file mode
        rotate_file(input_path, output_path, rotations)
        print(f"Rotated graph {args.times}x90° and saved to {output_path}")


if __name__ == "__main__":
    main()
