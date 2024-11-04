
import sys
import minizinc
import time
from utils import utils

def main(file_path):
    start_time = time.time()
    print(f'Using the instance {file_path}\n')  
    
    # # Create a MiniZinc model
    model = minizinc.Model("models/MCP.mzn")


    # # Transform Model into a instance
    gecode = minizinc.Solver.lookup("gecode")
    inst = minizinc.Instance(gecode, model)

    m, n, max_load, weights, distances = utils.read_dat_file(file_path, print_summary = True)

    inst["m"] = m
    inst["n"] = n
    inst["max_load"] = max_load
    inst["weights"] = weights
    inst["distances"] = distances


    # # Solve the instance
    result = inst.solve()

    if result.solution:
        print("Optimal Solution:")
        print(result.solution)
        print("")
        # print(result.solution.assign)
    else:
        print("No solution found.")
    stop_time = time.time()
    print(f"execution time: {stop_time-start_time:.2f} sec")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python cp.py <path_to_data_file>")
        sys.exit(1)
    if len(sys.argv) == 2:
        file_path = 'instances/' + sys.argv[1]
    else: 
        file_path = 'instances/inst01.dat'

    main(file_path)




