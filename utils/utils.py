import numpy as np
import os
import json
from datetime import datetime
import time
import argparse
import multiprocessing
import sys
import signal

def signal_handler(sig, frame):
    print("Terminating processes and exiting...")
    for p in multiprocessing.active_children():
        p.terminate()
    sys.exit(0)

def current_datetime():
    """ Prints the current date and time in the format YYYY-MM-DD HH:MM:SS """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def log(message, verbose=True):
    """ Logs a message with the option to control verbosity.

        Args:
            message (str): The message to be logged.
            verbose (bool): If True, the message is printed to the console.
    """
    if verbose:
        print(f"[LOG] {message}")

def save_results(res_path, inst_id, solver_name, result_data):
    file_path = os.path.join(res_path, f"{inst_id}.json")
    log(file_path)
    with open(file_path, "w") as f:
        json.dump({solver_name: result_data}, f, indent=4)
    log(f"JSON data saved to {file_path}")

def check_elapsed_time(start_time,timeout,print_time=False):
    """ Checks if the elapsed time has exceeded a specified timeout.

        Args:
            start_time (float): The time at which the process started.
            timeout (float): The timeout threshold in seconds.
            print_time (bool): If True, the elapsed time is printed.

        Returns:
            bool: True if elapsed time exceeds timeout, otherwise False.
    """
    elapsed_time = time.time() - start_time
    if print_time:
        log(f"elapsed time: {round(elapsed_time,2)}")
    return elapsed_time > timeout

# Function to read the .dat file and store first four lines
def read_dat_file(file_path, print_summary = False):
    """ Reads a .dat file and extracts the first four lines as well as the rest of the data.

        Args:
            file_path (str): Path to the .dat file to be read.
            print_summary (bool): If True, a summary of the contents is printed.

        Returns:
            tuple: A tuple containing:
                - m (int): Number of couriers.
                - n (int): Number of items.
                - load_sizes (list): List of load sizes for each courier.
                - item_sizes (list): List of item sizes.
                - data (np.array): The distance matrix as a NumPy array.
    """
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
    """ Encodes the input by sorting the items and couriers based on specified flags and returns updated input data.

        Args:
            item_sizes (list): A list of item sizes.
            load_limits (list): A list of courier load limits.
            distance_matrix (list of lists): The distance matrix between items and couriers.
            sort_items (bool): If True, sorts the items based on their size.
            sort_loads (bool): If True, sorts couriers based on their load limits.

        Returns:
            tuple: A tuple containing:
                - item_sizes (list): The sorted item sizes.
                - load_limits (list): The sorted load limits.
                - distance_matrix (list of lists): The updated distance matrix after sorting.
                - (item_encodings, sorted_indices_load): A tuple with item encodings and sorted load indices.
    """
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

def process_instances_input(inst_path, res_path, selected_instances, folder_name, solver_function, solver_name):
    """ Processes instances from a specified input folder and saves the results to the output folder.

        Args:
            inst_path (str): Path to the folder containing instance files.
            res_path (str): Path to the folder where results will be stored.
            selected_instances (list): List of instance IDs to process.
            folder_name (str): Name of the subfolder in which results will be saved.
            solver_function (function): The solver function to process each instance.
            solver_name (str): The name of the solver to be included in the output file.
    """
    signal.signal(signal.SIGINT, signal_handler)  # Catch Ctrl+C
    for filename in os.listdir(inst_path):
        inst_file = os.path.join(inst_path, filename)
        inst_id = int(filename[4:-4])
        if inst_id in selected_instances:
            n_couriers, n_items, capacity, sizes, dist_matrix = read_dat_file(inst_file)
            log(f'\tLoaded input instance {filename}')
            process = multiprocessing.Process(
                target=solver_function,
                args=(n_couriers, n_items, capacity, sizes, dist_matrix, res_path, inst_id, solver_name)
            )
            process.start()
            process.join(timeout=300)
            if process.is_alive():
                log("Timeout reached, terminating!")
                process.terminate()  # Forcefully terminate the process
                process.join()  # Ensure cleanup
                result = {
                    "time": 300,
                    "optimal": False,
                    "obj": 0,  # Set to 0 if no objective found
                    "sol": []
                }

                save_results(res_path, inst_id, solver_name, result)

def parsing_arguments(program): 
    
    if program == 'cp':
        parser = argparse.ArgumentParser(
            description="Solve problems with specified solver and instances."
        )

        parser.add_argument(
            '--mode',
            choices=['normal', 'superuser'],
            default='normal',
            help="Specify which mode to run."
        )

        # Add the --models flag
        parser.add_argument(
            '--models',
            choices=['all', 'sym', 'lns', 'plain', 'custom'],
            default='all',
            help="Specify the model to use. Options: all, sym, lns, plain, custom."
        )

        # Add the --instances flag
        parser.add_argument(
            '--instances',
            choices=['all', 'soft', 'hard'],
            default="all",
            help="Specify the instances to process. Options: all, soft, hard."
        )

        # Add the --solvers flag
        parser.add_argument(
            '--solvers',
            choices=['all', 'gecode', 'chuffed'],
            default='all',
            help="Specify the solver to use. Options: all, gecode, chuffed."
        )

        # Add the --save flag
        parser.add_argument(
            '--save',
            choices=['true', 'false'],
            default='true',
            help="Specify whether you want the results saved in a JSON."
        )

        parser.add_argument(
            '--results',
            default='results',
            help="Specify where do you want to save the results."
        )

        # Disallow unrecognized arguments
        args_cp = parser.parse_args()

        return args_cp