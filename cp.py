import sys
import minizinc
import time
import datetime
import os
import math
import json
import numpy as np
from utils import utils

def build_solution(res):
    packages = res["packages"]
    path = res["path"]
    sol = list()

    for c, package in enumerate(packages):
        items = list()
        count = np.count_nonzero(package)
        index = len(package) # starting from depot

        for _ in range(count) :
            items.append(path[c][index])
            index = path[c][index] - 1 #  -1 because index starts from 0
        sol.append(items)

    return sol

def solve_instance(model_path, instance_path, solver, time_limit):
    print(f"\nUsing: {model_path}\nSolving: {instance_path}\nSolver: {solver}")
    start_time = time.time()

    error = False
    
    model = minizinc.Model(model_path)
    solver = minizinc.Solver.lookup(solver)
    inst = minizinc.Instance(solver, model)

    m, n, max_load, weights, distances = utils.read_dat_file(instance_path, print_summary = False)

    inst["m"] = m
    inst["n"] = n
    inst["max_load"] = max_load
    inst["weights"] = weights
    inst["distances"] = distances

    try:
        res = inst.solve(timeout=time_limit)
    except Exception as e:
        print(f"Error: {e}")
        error = True
        print("[ERR] No solution found.")
         
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    solving_time, optimal, obj, sol = None, None, None, None

    if error or res.status == minizinc.Status.UNKNOWN:
        if not error:
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

    else:
        obj = res["objective"] 

        sol = build_solution(res)
        
        if res.status == minizinc.Status.SATISFIED:
            print(f"[SAT] Solution found! obj = {obj}")
            solving_time = 300
            optimal = False
            
        elif res.status == minizinc.Status.OPTIMAL_SOLUTION:
            print(f"[OPT] Solution found! obj = {obj}")
            solving_time = math.floor(elapsed_time)
            optimal = True

    result = {
        "time": solving_time,
        "optimal": optimal,
        "obj": obj,
        "sol": sol
    }
    

    return result, round(elapsed_time,2)


def save_results(res_path, inst_id, solver, model_name, result):

    os.makedirs(res_path, exist_ok=True)
    output_file_path = os.path.join(res_path, f"{inst_id}.json")
    configuration = solver + '_' + model_name

    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    existing_data[configuration] = result

    with open(output_file_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

    print(f"data saved ---> {output_file_path}")


def main(model_folder, selected_instances, result_folder, solvers, save, time_limit=300):
    res_path = f"{result_folder}/CP"
    instance_folder = "instances"
    time_limit = datetime.timedelta(seconds=int(time_limit))

    if save:
        print(f"Results saving ENABLED on {res_path}")
    else:
        print("Results saving DISABLED.")

    for model_name in os.listdir(model_folder):
        model_path = os.path.join(model_folder, model_name)

        if os.path.isfile(model_path):
            for solver in solvers:
                for instance_name in os.listdir(instance_folder):
                    instance_path = os.path.join(instance_folder, instance_name)
                    inst_id = int(instance_name[4:-4])
                    
                    if inst_id in selected_instances:
                        if solver == "gecode" or ("lns" not in model_name): 
                            if os.path.isfile(instance_path):
                        
                                result, elapsed_time = solve_instance(model_path, instance_path, solver, time_limit)
                            
                                print(f"Elapsed time: {elapsed_time} sec.")

                                if save:

                                    utils.save_results(res_path, inst_id, solver, result, model_name)
    print("\nDone!")


def handle_args(args):
    if args.models == 'custom':
        model = "models_CP/all/" + input("Type the name of the model you want to use: ")
        while(not os.path.isfile(model)):
            model = "models_CP/all/" + input(f"\n{model} doesn't exist! Try again: ")
        os.system(f"rm -r models_CP/custom/*") # remove previous custom models
        os.system(f"cp {model} models_CP/custom/") # copy the selected model in the custom folder
        
        model_folder = "models_CP/custom"

    else:
        model_folder = f"models_CP/{args.models}"
    
    if args.instances.isdigit():
        selected_instances = [int(args.instances)]
    else:
        selected_instances = list(map(int, args.instances.strip("[]").split(",")))

    results_folder = args.results
    solvers = ['gecode','chuffed'] if args.solvers == 'all' else [args.solvers]
    save = args.save 
    time_limit = args.time_limit

    return model_folder, selected_instances, results_folder, solvers, save, time_limit


if __name__ == "__main__":

    model_folder, selected_instances, results_folder, solvers, save, time_limit = handle_args(utils.parsing_arguments(program='cp'))
    
    print(f"Model folder: {model_folder}")
    print(f"Selected instances: {selected_instances}")
    print(f"Results folder: {results_folder}")
    print(f"Solvers: {solvers}")
    print(f"Save results: {save}")
    print(f"Time limit: {time_limit}")

    main(model_folder, selected_instances, results_folder, solvers, save, time_limit)

    # HELP

    # cp.py accepts the following optional commands:

    # --models: 
    # choices=['all', 'sym', 'lns', 'plain'],
    # default="all",
    # help="Specify the cp models to use."

    # --instances: 
    # choices=['all', 'soft', 'hard'],
    # default="all",
    # help="Specify the instances to process."

    # --solvers
    # choices=['all', 'gecode', 'chuffed'],
    # default="all",
    # help="Specify the solver to use."

    # --save
    # choices=['true', 'false'],
    # default='true',
    # help="Specify whether you want the results saved in a JSON."

    # --results 
    # default='results',
    # help="Specify where do you want to save the results."

    # --mode
    # choices=['normal', 'superuser'],
    # default='normal',
    # help="Specify which mode to run. If 'superuser' is selected, no need to specify models, instances and solvers."






