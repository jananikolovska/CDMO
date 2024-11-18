from z3 import *
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import time
from math import ceil
import json
from itertools import permutations
from utils.utils import process_instances_input, current_datetime
#from utils.smt import solve_mcp_z3
import random

def round_trip_distance(item, depot, distance_matrix):
    return distance_matrix[depot][item] + distance_matrix[item][depot]

def check_elapsed_time(start_time,timeout,print_time=True):
    elapsed_time = time.time() - start_time
    if print_time:
        print(f"elapsed time: {elapsed_time}")
    return elapsed_time > timeout

def compute_larger_bound(matrix):
    from heapq import nlargest

    n = len(matrix)
    used_columns = set()
    total_sum = 0

    for row in matrix:
        # Find the largest valid (unused column) weights in this row
        best_values = [(value, col) for col, value in enumerate(row) if col not in used_columns]
        if best_values:
            largest_value, col = max(best_values)
            total_sum += largest_value
            used_columns.add(col)

    return total_sum


# Compute the lower bound as the max of the round-trip distances for all items
def compute_lower_bound(depot, num_items, distance_matrix):
    max_distance = 0
    for item in range(num_items):
        # Calculate the round-trip distance for each item
        round_trip_dist = round_trip_distance(item, depot, distance_matrix)
        max_distance = max(max_distance, round_trip_dist)

    return max_distance

def solve_mcp_z3(num_couriers, num_items, load_limits, item_sizes, int_distance_matrix, verbose=False):
    # Verbose logging function
    def log(message):
        if verbose:
            current_datetime()
            print(f"[LOG] {message}")

    # Initialize the Z3 optimizer
    opt = Optimize()

    # Set a timeout (in milliseconds)
    timeout = 300  # 300 seconds

    log("Initialization complete. Setting up variables and constraints...")

    start_time = time.time()
    # Sort the items
    sorted_indices = sorted(range(len(item_sizes)), key=lambda k: item_sizes[k])
    item_sizes = [item_sizes[i] for i in sorted_indices]
    sorted_indices = sorted_indices + [len(item_sizes)]

    # Sort the distance matrix rows and columns according to the sorted items
    int_distance_matrix = [[int_distance_matrix[i][j] for j in sorted_indices] for i in sorted_indices]

    # Sort couriers by capacity (ascending)
    load_limits.sort()

    distance_matrix = [[IntVal(int_distance_matrix[p][q]) for q in range(num_items + 1)]
                       for p in range(num_items + 1)]

    # Decision Variables:
    # x[i][j] = 1 if courier i is assigned to item j, 0 otherwise
    x = [[Bool(f'x_{i}_{j}') for j in range(num_items)] for i in range(num_couriers)]

    # y[i][p][q] is now a bitvector instead of a bool matrix
    # y[i][p][q] = 1 if courier i travels from point p to point q, 0 otherwise
    y = [[[BitVec(f'y_{i}_{p}_{q}', 1) for q in range(num_items + 1)]
          for p in range(num_items + 1)] for i in range(num_couriers)]

    # d[i] = total distance traveled by courier i
    d = [Int(f'd_{i}') for i in range(num_couriers)]

    # max_distance = the maximum distance traveled by any courier
    max_distance = Int('max_distance')


    # Initialize lower bound
    depot = num_items
    upper_bound = compute_larger_bound(int_distance_matrix)
    lower_bound = compute_lower_bound(depot, num_items, int_distance_matrix)
    opt.add(max_distance <= upper_bound)
    opt.add(max_distance >= lower_bound)
    log(f"Bounds set. Upper: {upper_bound}, Lower: {lower_bound}")


    # Breaking courier symmetry - Enforce order by distance traveled
    for i in range(num_couriers - 1):
        opt.add(d[i] <= d[i + 1])

    if check_elapsed_time(start_time, timeout):
        print('1 Elapsed time too much')
        return

    # Constraints:
    log("Adding constraints...")
    # Each item must be assigned to exactly one courier DEMAND FULFILLMENT
    for j in range(num_items):
        opt.add(Sum([If(x[i][j], 1, 0) for i in range(num_couriers)]) == 1)

    if check_elapsed_time(start_time, timeout):
        print('2 Elapsed time too much')
        return

    # Each courier's load must not exceed their capacity CAPACITY CONSTRAINT
    for i in range(num_couriers):
        # Add the load constraint for each courier
        opt.add(Sum([If(x[i][j], item_sizes[j], 0) for j in range(num_items)]) <= load_limits[i])

    if check_elapsed_time(start_time, timeout):
        print('3 Elapsed time too much')
        return

    # Early Exclusion of Unusable Couriers
    for i in range(num_couriers):
        for j in range(num_items):
            if item_sizes[j] > load_limits[i]:
                opt.add(Not(x[i][j]))
                for k in range(j + 1, num_items):
                    opt.add(Not(x[i][k]))

    if check_elapsed_time(start_time, timeout):
        print('4 Elapsed time too much')
        return

    # Ensure each courier has at least one item
    for i in range(num_couriers):
        opt.add(Sum([x[i][j] for j in range(num_items)]) >= 1)

    if check_elapsed_time(start_time, timeout):
        print('4.5 Elapsed time too much')
        return

    # Add constraint to prevent staying at the same location
    for i in range(num_couriers):
        for p in range(num_items + 1):
            opt.add(y[i][p][p] == 0)  # Ensure no self-loop (y[i][p][p] = 0)

    if check_elapsed_time(start_time, timeout):
        print('5 Elapsed time too much')
        return

    # Tour constraints: each courier starts and ends at the origin
    # for i in range(num_couriers):
    #     opt.add(Sum([If(y[i][num_items][q] == 1, 1, 0) for q in range(num_items)]) == 1)  # Start at origin
    #     opt.add(Sum([If(y[i][p][num_items] == 1, 1, 0) for p in range(num_items)]) == 1)  # End at origin
    # Start and end at origin using bitwise encoding
    # Heule encoding: ensure that at most one y[i][p][q] is 1
    # Heule encoding: ensure that at most one y[i][p][q] is 1
    for q in range(num_items-1):
        # Use bitwise OR to combine all y[i][10][q]
        combined_start = (y[0][num_items][q]==1)  # Start with the first element
        for i in range(1, num_couriers):
            combined_start = Or(combined_start, y[i][p][q]==1)  # Bitwise OR across all i for the same q

        # Add the constraint that the sum of y[i][10][q] across all i is <= 1
        opt.add(If(combined_start, 1, 0) <= 1)

    for p in range(num_items-1):
        # Use bitwise OR to combine all y[i][10][q]
        combined_end = (y[0][p][num_items]==1)  # Start with the first element
        for i in range(1, num_couriers):
            combined_end = Or(combined_end, y[i][p][q]==1)  # Bitwise OR across all i for the same q

        opt.add(If(combined_end, 1, 0) <= 1)


    if check_elapsed_time(start_time, timeout):
        print('6 Elapsed time too much')
        return

    # Route Continuity
    for i in range(num_couriers):
        for p in range(num_items):  # Excluding depot
            for q in range(num_items):  # Excluding depot
                if p != q:
                    # If y[i][p][q] is True, y[i][q][p] should be False (no return trips)
                    opt.add(Implies(y[i][p][q] == 1, y[i][q][p] == 0))

    # Ensure no revisits to the same location for each courier
    for i in range(num_couriers):
        for p in range(num_items):
            opt.add(And([
                Implies(And(y[i][p][q] == 1, y[i][q][r] == 1), y[i][r][p] == 0)
                for q in range(num_items) for r in range(num_items)
                if q != p and r != p and q != r
            ]))

    if check_elapsed_time(start_time, timeout):
        print('7 Elapsed time too much')
        return

    # Route Continuity: Only include y[i][p][q] if x[i][q] is true (i.e., courier i is assigned to item q)
    for i in range(num_couriers):
        for p in range(num_items + 1):  # Including depot
            for q in range(num_items):  # Exclude depot as it's only the starting/ending point
                opt.add(Implies(y[i][p][q] == 1, x[i][q]))

    if check_elapsed_time(start_time, timeout):
        print('8 Elapsed time too much')
        return

    # Preventing redundant "revisits" to a location by ensuring only allowed transitions
    for i in range(num_couriers):
        for j in range(num_items):
            opt.add(Sum([If(y[i][p][j] == 1, 1, 0) for p in range(num_items + 1)]) == If(x[i][j], 1, 0))
            opt.add(Sum([If(y[i][j][q] == 1, 1, 0) for q in range(num_items + 1)]) == If(x[i][j], 1, 0))

    if check_elapsed_time(start_time, timeout):
        print('9 Elapsed time too much')
        return

    # Calculate total distance for each courier
    for i in range(num_couriers):
        # Create a list to hold distance terms for the current courier
        distance_terms = []

        # Loop over all points (including depot) to sum distances
        for p in range(num_items + 1):  # Including depot
            for q in range(num_items + 1):  # Including depot
                # Use an integer value for distance matrix
                distance = distance_matrix[p][q]  # Make sure this is an integer
                distance_terms.append(If(y[i][p][q] == 1, distance, 0))

        # Add the constraint for the total distance for courier i
        opt.add(d[i] == Sum(distance_terms))

    # The maximum distance is at least the distance traveled by any courier
    for i in range(num_couriers):
        opt.add(max_distance >= d[i])

    # Objective: minimize the maximum distance
    opt.minimize(max_distance)
    log("Starting solver...")
    opt.set("timeout", (timeout - int(time.time() - start_time)) * 1000)
    log(f"Timeout: {timeout - int(time.time() - start_time)}")
    current_datetime()

    # Solve the problem with a timer
    result = opt.check()
    runtime = time.time() - start_time

    optimal = False
    obj = None
    routes = []

    if result == sat or result == unknown:
        if result == unknown:
            opt.reason_unknown()
        model = opt.model()
        # If the result is sat, we assume it's optimal
        if result == sat:
            optimal = True
        else:
            optimal = False  # Unknown means timeout reached without proving optimality

        # Extract the objective value
        if model.eval(max_distance, model_completion=True) is not None:
            obj = model[max_distance].as_long()

        # Extract routes for each courier
        for i in range(num_couriers):
            assigned_items = [j + 1 for j in range(num_items) if is_true(model.evaluate(x[i][j]))]
            routes.append(assigned_items)
            log(f'Courier {i + 1}: Assigned items: {assigned_items}, Distance: {model[d[i]]}')

        # Optional: Print y and x variables for the first courier (for debugging)
        # courier_id = 1  # Change as needed
        # for p in range(num_items + 1):
        #     for q in range(num_items + 1):
        #         y_val = model.evaluate(y[courier_id][p][q], model_completion=True)
        #         print(f"y[{courier_id}][{p}][{q}] = {y_val}")
        # for p in range(num_items):
        #     x_val = model.evaluate(x[courier_id][p], model_completion=True)
        #     print(f"x[{courier_id}][{p}] = {x_val}")
    else:
        log("No solution found.")

    exec_time = min(int(runtime), timeout)

    # Construct the JSON result
    result = {
        "time": exec_time,
        "optimal": optimal,
        "obj": obj if obj is not None else 0,  # Set to 0 if no objective found
        "sol": routes,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound
    }

    return result

if __name__ == "__main__":
    args = sys.argv
    print('SAT solver started')
    inst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[1])
    os.makedirs(args[2], exist_ok=True)
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[2])
    process_instances_input(inst_path,res_path,solve_mcp_z3,"try_without_optimal",verbose=True)
    print('SAT solver done')

