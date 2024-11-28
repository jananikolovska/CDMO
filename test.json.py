import json
import os

for json_file in os.listdir("results/CP"):
    with open("results/CP/" + json_file, 'r') as file:
        data = json.load(file)
        print(f"File: {json_file}")
        # keys = [key for key in all_keys if "lms" in key]
        keys = list(data.keys())
        obj_values = [data[key]["obj"] for key in keys]
        print(f"obj_values = {obj_values}")
        # select the key associated to the best obj value excluding 0 values
        best_key = keys[obj_values.index(min([obj for obj in obj_values if obj > 0]))]
        print(f"best_key = {best_key}")
        print(f"best_obj = {data[best_key]['obj']}")
        print("\n\n")