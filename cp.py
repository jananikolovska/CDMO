
import sys
import minizinc
import time
import datetime
import os
import math
import json
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
    res_path = "results/CP"
    time_limit = datetime.timedelta(seconds=time_limit)
    # model_path = "models/MCP v1.1.1.mzn"
    solvers = ["gecode", "chuffed"]
    for model_name in os.listdir(model_folder):
        print(f"Using model {model_name}")
        model_path = os.path.join(model_folder, model_name)
        if os.path.isfile(model_path):
            for instance_name in os.listdir(instance_folder):
                instance_path = os.path.join(instance_folder, instance_name)
                inst_id = int(instance_name[4:-4])
                if os.path.isfile(instance_path):
                    print(f"Solving instance {instance_name}")
                    for solver in solvers:
                        print(f"Using solver {solver}")
                        result, elapsed_time = solve_instance(model_path, instance_path, solver, time_limit)
                        print(f"results: {result}")
                        print(f"Elapsed time: {elapsed_time} sec.")

                        # output result in a json file
                        output_file_path = os.path.join(res_path, f"{inst_id}.json")
                        configuration = solver+'_'+model_name

                        # # Save the JSON data to the file
                        # with open(output_file_path, "w") as json_file:
                        #     json.dump({configuration: result}, json_file, indent=4)

                        if os.path.exists(output_file_path):
                            # Load existing data
                            with open(output_file_path, "r") as json_file:
                                existing_data = json.load(json_file)
                        else:
                            # If file doesn't exist, start with an empty dictionary
                            existing_data = {}

                        # Update the data with the new configuration and result
                        existing_data[configuration] = result

                        # Save the updated data back to the file
                        with open(output_file_path, "w") as json_file:
                            json.dump(existing_data, json_file, indent=4)

                        print(f"\tJSON data saved to {output_file_path}")
                        print("")
    
    print("Done!")




if __name__ == "__main__":

    model_path = "models"
    instance_path = "instances"
    time_limit = 300
    main(model_path, instance_path, time_limit)





