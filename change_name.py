import os
import json


index = 0
folder_input = ["pattern","./pattern_old"]
folder_output = "./pattern_merge"

os.makedirs(folder_output, exist_ok=True)
for folder in folder_input:
    for file in os.listdir(folder):
        index += 1
        if file.endswith(".json"):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            data["metadata"]["name"] = file.split(".")[0]
            with open(os.path.join(folder_output, str(index)+".json"), 'w') as f:
                json.dump(data, f)
