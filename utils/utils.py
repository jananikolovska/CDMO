import numpy as np
import os
import json
from datetime import datetime
import time
import argparse

def current_datetime():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def log(message, verbose=True):
    if verbose:
        print(f"[LOG] {message}")

def check_elapsed_time(start_time,timeout,print_time=False):
    elapsed_time = time.time() - start_time
    if print_time:
        log(f"elapsed time: {round(elapsed_time,2)}")
    return elapsed_time > timeout

# Function to read the .dat file and store first four lines
def read_dat_file(file_path, print_summary = False):
    with open(file_path, 'r') as file:
        # Read the first four lines
        m = int(file.readline().strip())  # First line
        n = int(file.readline().strip())  # Second line
        load_sizes = list(map(int, file.readline().strip().split()))  # Third line
        item_sizes = list(map(int, file.readline().strip().split()))  # Fourth line

        # Read the rest of the data into a list (if needed)
        data = []
        for line in file:
            data.append(list(map(int, line.strip().split())))

    if print_summary:
        print("############SUMMARY############")
        print(f"num_couriers: {m}")
        print(f"num_items: {n}")
        print(f"Load sizes: {load_sizes}")
        print(f"Item sizes: {item_sizes}")
        print("Distance matrix:")
        print(data)
        print("############END############\n")

    return m, n, load_sizes, item_sizes, np.array(data)

def encode_input(item_sizes, load_limits , distance_matrix, sort_items, sort_loads):
    if sort_items:
        # Sort the items
        sorted_indices_item = sorted(range(len(item_sizes)), key=lambda k: item_sizes[k])
        item_sizes = [item_sizes[i] for i in sorted_indices_item]
        sorted_indices_item = sorted_indices_item + [len(item_sizes)]
        item_encodings = dict(zip(list(range(1, len(item_sizes) + 1)), [inx + 1 for inx in sorted_indices_item]))

        # Sort the distance matrix rows and columns according to the sorted items
        distance_matrix = [[distance_matrix[i][j] for j in sorted_indices_item] for i in sorted_indices_item]

    if sort_loads:
        # Sort couriers by capacity (ascending)
        sorted_indices_load = sorted(range(len(load_limits)), key=lambda k: load_limits[k])
        load_limits = [load_limits[i] for i in sorted_indices_load]

        return item_sizes, load_limits , distance_matrix, (item_encodings,sorted_indices_load)

def process_instances_input(inst_path, res_path, solver_function,solver_name):
    for filename in os.listdir(inst_path):
        inst_file = os.path.join(inst_path, filename)
        inst_id = int(filename[4:-4])
        n_couriers, n_items, capacity, sizes, dist_matrix = read_dat_file(inst_file)
        log(f'\tLoaded input instance {filename}')
        result = solver_function(n_couriers, n_items, capacity, sizes, dist_matrix)
        file_path = os.path.join(res_path, f"{inst_id}.json")

        # Save the JSON data to the file
        with open(file_path, "w") as json_file:
            json.dump({solver_name: result}, json_file, indent=4)

        log(f"JSON data saved to {file_path}")

def parse_args(args):
    def parse_instances(value):
        # Check if the value is "all" or a comma-separated list
        if value.lower() == "all":
            return "all"
        return value.split(",")  # Split comma-separated values into a list


    parser = argparse.ArgumentParser(
        description="Solve problems with specified solver and instances."
    )

    # Add the --models flag
    parser.add_argument(
        '--models',
        choices=['all', 'sym', 'lns', 'plain', 'custom'],
        required=True,
        default='all',
        help="Specify the solver to use. Options: all, gecode, chuffed."
    )

    # Add the --solver flag
    parser.add_argument(
        '--solver',
        choices=['all', 'gecode', 'chuffed'],
        required=True,
        default='all',
        help="Specify the solver to use. Options: all, gecode, chuffed."
    )

    # Add the --instances flag
    parser.add_argument(
        '--instances',
        type=parse_instances,
        default='all',
        help="Specify the instances to process. Options: a, b, c."
    )

    # Disallow unrecognized arguments
    args = parser.parse_args()

    # Print arguments (for demonstration purposes)
    print(f"Solver: {args.solver}")
    print(f"Models: {args.models}")
    print(f"Instances: {args.instances}")
    return args.models, args.solver, args.instances