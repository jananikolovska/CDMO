
import sys
import minizinc
import time
import datetime
import os
import math
from utils import utils

def solve_instance(model_path, instance_path, solver, time_limit):
    print(f"Using: {model_path} for solving the instance {instance_path}")
    start_time = time.time()
    
    model = minizinc.Model(model_path)
    solver = minizinc.Solver.lookup(solver)
    inst = minizinc.Instance(solver, model)

    m, n, max_load, weights, distances = utils.read_dat_file(instance_path, print_summary = False)

    inst["m"] = m
    inst["n"] = n
    inst["max_load"] = max_load
    inst["weights"] = weights
    inst["distances"] = distances

    res = inst.solve(timeout=time_limit)
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    solving_time, optimal, obj, sol = None, None, None, None
    print(f"STATUS = {res.status}")
    if res.status == minizinc.Status.UNKNOWN:
        print("[UNK] Time limit reached. No solution found.")
        solving_time = 300
        optimal = False
        obj = 0
        sol = None
        
    elif res.status == minizinc.Status.UNSATISFIABLE:
        print("[UNSAT] Unsatisfiable. Error in the model (?)")
        solving_time = math.floor(elapsed_time)
        optimal = False
        obj = 0
        sol = None
    # if a solution is found:
    else:
        obj = res["objective"] 
        sol = list()
        for package in res["packages"]:
            sol.append([i+1 for i in range(len(package)) if package[i] == 1])
        
        if res.status == minizinc.Status.SATISFIED:
            print(f"[SAT] Solution found! obj = {obj}")
            print(f"packages = {res['packages']}")
            solving_time = 300
            optimal = False
            
        elif res.status == minizinc.Status.OPTIMAL_SOLUTION:
            print(f"[OPT] Solution found! obj = {obj}")
            print(f"packages = {res['packages']}")
            solving_time = math.floor(elapsed_time)
            optimal = True

    result = {
        "time": solving_time,
        "optimal": optimal,
        "obj": obj,
        "sol": sol
    }
    

    return result, round(elapsed_time,2)

def main(model_folder, instance_folder, time_limit):
    time_limit = datetime.timedelta(seconds=time_limit)
    # model_path = "models/MCP v1.1.1.mzn"
    solvers = ["gecode", "chuffed"]
    for model_name in os.listdir(model_folder):
        model_path = os.path.join(model_folder, model_name)
        # inst_id = int(filename[4:-4])
        if os.path.isfile(model_path):
            for instance_name in os.listdir(instance_folder):
                instance_path = os.path.join(instance_folder, instance_name)
                # inst_id = int(instance_name[4:-4])
                if os.path.isfile(instance_path):
                    for solver in solvers:
                        result, elapsed_time = solve_instance(model_path, instance_path, solver, time_limit)
                        print(f"results: {result}")
                        print(f"Elapsed time: {elapsed_time} sec.")



if __name__ == "__main__":
    # if len(sys.argv) > 2:
    #     print("Usage: python cp.py <path_to_data_file>")
    #     sys.exit(1)
    # if len(sys.argv) == 2:
    #     instance_path = 'instances/' + sys.argv[1]
    # else: 
    #     instance_path = 'instances/inst01.dat'
    model_path = "models"
    instance_path = "instances"
    time_limit = 5
    main(model_path, instance_path, time_limit)





