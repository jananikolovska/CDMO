import sys
import minizinc
import time
import datetime
import os
import math
import json
from utils import utils

def solve_instance(model_path, instance_path, solver, time_limit):
    print(f"\nUsing: {model_path}\nSolving: {instance_path}\nSolver: {solver}")
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

    else:
        obj = res["objective"] 
        sol = list()
        for package in res["packages"]:
            sol.append([i+1 for i in range(len(package)) if package[i] == 1])
        
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

def main(model_folder, instance_folder, solvers, save_results = True, time_limit=300):
    res_path = "results/CP/try"
    time_limit = datetime.timedelta(seconds=int(time_limit))

    if save_results:
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

                    if os.path.isfile(instance_path):
                        result, elapsed_time = solve_instance(model_path, instance_path, solver, time_limit)
                        print(f"Elapsed time: {elapsed_time} sec.")

                        if save_results:

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
    print("\nDone!")


def handle_args(args):
    if args == []:
        model_option = "all"
        instance_option = "all"
        solver_option = "all"

    elif len(args) != 3:
        raise ValueError("Invalid arguments. \n\
            Usage: python3 cp.py <models> <instances> <solvers> <save_results>.\n\
                <models>: all, sym, lns, plain, custom.\n\
                    <instances>: all, soft, hard.\n\
                        <solvers>: all, gecode, chuffed.")
    else:                    
        model_option = args[0]
        instance_option = args[1]
        solver_option = args[2]

    if model_option == "all":
        model_folder = "models/all"
    
    elif model_option == "sym":
        model_folder = "models/sym"

    elif model_option == "lns":
        model_folder = "models/lns"

    elif model_option == "plain":
        model_folder = "models/plain"   

    elif model_option == "custom":
        model_folder = "models/custom"  

    else:
        raise ValueError("Invalid model option. \n\
            Usage: python3 cp.py <models> <instances> <solvers>.\n\
                Please choose one of the following <models>: all, sym, lns, plain, custom.")

    if instance_option == "all":
        instance_folder = "instances_CP/all"

    elif instance_option == "soft":
        instance_folder = "instances_CP/soft"
    
    elif instance_option == "hard":
        instance_folder = "instances_CP/hard"

    elif instance_option == "custom":
        instance_folder = "instances_CP/custom"

    else:
        raise ValueError("Invalid instance option. \n\
            Usage: python3 cp.py <models> <instances> <solvers>.\n\
                Please choose one of the following <instances>: all, soft, hard, custom.")
    
    if solver_option == "all":
        solvers = ["gecode","chuffed"]
    elif solver_option == "gecode":
        solvers = ["gecode"]
    elif solver_option == "chuffed":
        solvers = ["chuffed"]
    else:
        raise ValueError("Invalid solver option. \n\
            Usage: python3 cp.py <models> <instances> <solvers>.\n\
                Please choose one of the following <solvers>: all, gecode, chuffed.")

    print(f"\nRunning {model_option} models on {instance_option} instances with {solver_option} solver(s)!")

    return model_folder, instance_folder, solvers

if __name__ == "__main__":

    model_folder, instance_folder, solvers = handle_args(sys.argv[1:])
    save_results = True
    main(model_folder, instance_folder, solvers, save_results)

    # the idea is to execute the program in this way:
    # python3 cp.py <models> <instances> <solvers> 
    # python3 cp.py all_models all_instances all_solvers
    # python3 cp.py sym_models soft_instances gecode
    # python3 cp.py lns_models hard_instances chuffed
    # python3 cp.py plain_models hard_instances gecode






