from z3 import *
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
from utils.utils import process_instances_input

def solve_mcp_z3(num_couriers, num_items, load_limits, item_sizes, int_distance_matrix):
    """
    Computes SAT best solutions
    :param num_couriers:
    :param num_items:
    :param load_limits:
    :param item_sizes:
    :param int_distance_matrix:
    :return:
    """
    distance_matrix = [[IntVal(int_distance_matrix[p][q]) for q in range(num_items+1)]
                   for p in range(num_items+1)]


    # Initialize the Z3 optimizer
    opt = Optimize()

    # Set a timeout of 300 seconds (300000 ms)
    opt.set("timeout", 300000)

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

    # Constraints:

    # Each item must be assigned to exactly one courier
    for j in range(num_items):
        opt.add(Sum([If(x[i][j], 1, 0) for i in range(num_couriers)]) == 1)

    # Each courier's load must not exceed their capacity
    for i in range(num_couriers):
      # Add the load constraint for each courier
      opt.add(Sum([If(x[i][j], item_sizes[j], 0) for j in range(num_items)]) <= load_limits[i])

    # Tour constraints: each courier starts and ends at the origin
    for i in range(num_couriers):
        opt.add(Sum([If(y[i][num_items][q], 1, 0) for q in range(num_items)]) == 1)  # Start at origin
        opt.add(Sum([If(y[i][p][num_items], 1, 0) for p in range(num_items)]) == 1)  # End at origin

    # Ensure each location is visited exactly once by the assigned courier
    for j in range(num_items):
        for i in range(num_couriers):
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
    optimal = False
    timeout = 300  # 300 seconds timeout

    # Check if the solution is feasible and minimize within the time limit
    if opt.check() == sat:
        model = opt.model()
        obj = model[max_distance].as_long()
        runtime = time.time() - start_time
        optimal = True
    else:
        runtime = timeout  # Max runtime if not solved optimally
        obj = None  # No objective found if unsolvable
        model = None

    # Solve the problem
    routes = []
    if model:
        for i in range(num_couriers):
            assigned_items = [j + 1 for j in range(num_items) if model.evaluate(x[i][j])]
            routes.append(assigned_items)
            print(f'Courier {i + 1}: Assigned items: {assigned_items}, Distance: {model[d[i]]}')
    else:
        print("No solution found.")

    exec_time = min(int(runtime), timeout)
    if exec_time >= timeout:
        optimal = False

    # Construct the JSON result
    result = {
        "time": exec_time,
        "optimal": optimal,
        "obj": obj if obj is not None else 0,  # Set to 0 if no objective found
        "sol": routes
    }

    # Return the result as a JSON object
    #return json.dumps(result)

    return result


if __name__ == "__main__":
    args = sys.argv
    print('SAT solver started')
    inst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[1])
    os.makedirs(args[2], exist_ok=True)
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[2])
    process_instances_input(inst_path,res_path,solve_mcp_z3)
    print('SAT solver done')

