from solver import get_3d_point_mesh
import numpy as np
import os

def build_off(file_path, output_path, folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            output_path = os.path.join("output_obj", folder_path, file.replace(".json", ".off"))
        vertices, faces = get_3d_point_mesh(file_path)

        if hasattr(vertices, "cpu"):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, "cpu"):
            faces = faces.cpu().numpy()

        vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("OFF\n")
            f.write(f"{len(vertices)} {len(faces)} 0\n")

            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        print("Saved:", output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build OFF file from json file")
    parser.add_argument("--input", default="output_9x9/output_sym_Y/2_5.json")
    parser.add_argument("--output", default="output_obj/2.5.off")
    parser.add_argument("--folder", default="output_9x9/output_sym_Y")
    args = parser.parse_args()

    build_off(args.input, args.output, args.folder)