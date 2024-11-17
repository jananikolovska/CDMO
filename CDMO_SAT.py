from z3 import *
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
from utils.utils import process_instances_input


def solve_mcp_z3(num_couriers, num_items, load_limits, item_sizes, int_distance_matrix):
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

    # Initialize the Z3 optimizer
    opt = Optimize()

    # Set a timeout (in milliseconds)
    timeout = 300 * 1000  # 300 seconds
    opt.set("timeout", timeout)

    # Decision Variables:
    # x[i][j] = 1 if courier i is assigned to item j, 0 otherwise
    x = [[Bool(f'x_{i}_{j}') for j in range(num_items)] for i in range(num_couriers)]

    # y[i][p][q] = 1 if courier i travels from point p to point q, 0 otherwise
    y = [[[Bool(f'y_{i}_{p}_{q}') for q in range(num_items + 1)]
          for p in range(num_items + 1)] for i in range(num_couriers)]

    # d[i] = total distance traveled by courier i
    d = [Int(f'd_{i}') for i in range(num_couriers)]

    # max_distance = the maximum distance traveled by any courier
    max_distance = Int('max_distance')

    # Symmetry breaking constraints

    # Breaking courier symmetry - Enforce order by distance traveled
    for i in range(num_couriers - 1):
        opt.add(d[i] <= d[i + 1])

    # # Proximity-priority heuristic: encourage closer matches
    # distance_weights = [[1 / (1 + distance_matrix[0][j]) for j in range(num_items)] for i in range(num_couriers)]
    # penalty_terms = []
    # for i in range(num_couriers):
    #     for j in range(num_items):
    #         penalty_terms.append(If(x[i][j], distance_weights[i][j], 0))
    # opt.minimize(Sum(penalty_terms))  # Minimize penalties as part of the objective

    # Constraints:

    # Each item must be assigned to exactly one courier DEMAND FULFILLMENT
    for j in range(num_items):
        opt.add(Sum([If(x[i][j], 1, 0) for i in range(num_couriers)]) == 1)

    # Each courier's load must not exceed their capacity CAPACITY CONSTRAINT
    for i in range(num_couriers):
        # Add the load constraint for each courier
        opt.add(Sum([If(x[i][j], item_sizes[j], 0) for j in range(num_items)]) <= load_limits[i])

    # Early Exclusion of Unusable Couriers
    for i in range(num_couriers):
        for j in range(num_items):
            if item_sizes[j] > load_limits[i]:
                opt.add(Not(x[i][j]))
                for k in range(j + 1, num_items):
                    opt.add(Not(x[i][k]))

    # Add constraint to prevent staying at the same location
    for i in range(num_couriers):
        for p in range(num_items + 1):
            opt.add(Not(y[i][p][p]))  # Ensure no self-loop

    # Tour constraints: each courier starts and ends at the origin
    for i in range(num_couriers):
        opt.add(Sum([If(y[i][num_items][q], 1, 0) for q in range(num_items)]) == 1)  # Start at origin
        opt.add(Sum([If(y[i][p][num_items], 1, 0) for p in range(num_items)]) == 1)  # End at origin

    # Route Continuity and No Revisits
    for i in range(num_couriers):
        for p in range(num_items):  # Excluding depot
            for q in range(num_items):  # Excluding depot
                if p != q:
                    # If y[i][p][q] is True, y[i][q][p] should be False (no return trips)
                    opt.add(Implies(y[i][p][q], Not(y[i][q][p])))
                # Ensures that each location p, once left, cannot be revisited by the same courier
                for r in range(num_items):
                    if r != p and r != q:
                        opt.add(Implies(And(y[i][p][q], y[i][q][r]), Not(y[i][r][p])))

    # Route Continuity: Only include y[i][p][q] if x[i][q] is true (i.e., courier i is assigned to item q)
    for i in range(num_couriers):
        for p in range(num_items + 1):  # Including depot
            for q in range(num_items):  # Exclude depot as it's only the starting/ending point
                opt.add(Implies(y[i][p][q], x[i][q]))

    # Preventing redundant "revisits" to a location by ensuring only allowed transitions
    for i in range(num_couriers):
        for j in range(num_items):
            opt.add(Sum([If(y[i][p][j], 1, 0) for p in range(num_items + 1)]) == If(x[i][j], 1, 0))
            opt.add(Sum([If(y[i][j][q], 1, 0) for q in range(num_items + 1)]) == If(x[i][j], 1, 0))

    # Calculate total distance for each courier
    for i in range(num_couriers):
        # Create a list to hold distance terms for the current courier
        distance_terms = []

        # Loop over all points (including depot) to sum distances
        for p in range(num_items + 1):  # Including depot
            for q in range(num_items + 1):  # Including depot
                # Use an integer value for distance matrix
                distance = distance_matrix[p][q]  # Make sure this is an integer
                distance_terms.append(If(y[i][p][q], distance, 0))

        # Add the constraint for the total distance for courier i
        opt.add(d[i] == Sum(distance_terms))

    # The maximum distance is at least the distance traveled by any courier
    for i in range(num_couriers):
        opt.add(max_distance >= d[i])

    # Objective: minimize the maximum distance
    opt.minimize(max_distance)

    # Solve the problem with a timer
    start_time = time.time()
    result = opt.check()
    runtime = time.time() - start_time

    optimal = False
    obj = None
    routes = []

    if result == sat or result == unknown:
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
            print(f'Courier {i + 1}: Assigned items: {assigned_items}, Distance: {model[d[i]]}')

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
        print("No solution found.")

    exec_time = min(int(runtime), timeout/1000)

    # Construct the JSON result
    result = {
        "time": exec_time,
        "optimal": optimal,
        "obj": obj if obj is not None else 0,  # Set to 0 if no objective found
        "sol": routes
    }

    # Return the result as a JSON object
    # return json.dumps(result)

    return result

if __name__ == "__main__":
    args = sys.argv
    print('SAT solver started')
    inst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[1])
    os.makedirs(args[2], exist_ok=True)
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[2])
    process_instances_input(inst_path,res_path,solve_mcp_z3,"try_without_optimal")
    print('SAT solver done')

