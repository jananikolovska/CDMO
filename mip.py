import argparse
import pulp as pl
import numpy as np
from utils import *
from utils1 import *
from math import floor
import json

class MIP_solver:

    __time_limit=0
    __instance=0

    #problem variables
    __m=0 #number of couriers
    __n=0 #number of items
    __D=[] #matrix of distances betweeen delivery points
    __L=[] #max load for each courier
    __S=[] #size of each item

    __n_loc=0 #number of locations in the problem
    __origin = 0 #index of the origin in the matrix 

    __solver =''
    __multiple_solvers=False

    #decision variables
    __lp_prob =''
    __x='' #3d array which is 1 if ...
    __courier_dist=''

    


    def __init__(self,res_path,inst_path, time_limit = 300  ) -> None:

        self.__time_limit=time_limit
        self.__res_path=res_path

        self.__lp_prob = LpProblem("Courier_Routing_Optimization", LpMinimize)
        self.__inst_path=inst_path

        #self.run_model()
            
    def __compute_upper_bound(self):
        ''''
        Computes a larger bound for the optimization problem by selecting the maximum values from the distance matrix
        
        Returns:
            int: The sum of the largest values from each row of the matrix.
        '''
        used_columns = set()
        total_sum = 0
        matrix = self.__D
        for row in matrix:
            best_values = [(value, col) for col, value in enumerate(row) if col not in used_columns]
            if best_values:
                largest_value, col = max(best_values)
                total_sum += largest_value
                used_columns.add(col)

        return total_sum
    

    def __compute_lower_bound(self):
        """
            Compute the maximum round-trip distance between the origin (n+1) and distribution points.
            :return: Maximum round-trip distance
        """
        # Number of distribution points (excluding the origin)
        n = len(self.__D) - 1
        origin_index = n  # Index of the origin (n+1 corresponds to 0-based index n)
        
        # Calculate round-trip distances
        max_distance = 0
        for i in range(n):  # Iterate through distribution points (0 to n-1)
            round_trip_distance = self.__D[origin_index][i] + self.__D[i][origin_index]
            max_distance = max(max_distance, round_trip_distance)
        
        return max_distance

        
    
    def __configure_problem(self):

        num_cities = self.__D.shape[0] - 1
        depot = num_cities + 1

        upper_bound = self.__compute_upper_bound()
        lower_bound = self.__compute_lower_bound()

        min_courier_dist = min(self.__D[num_cities, i] + self.__D[i, num_cities] for i in range(num_cities))


        # Decision variables
        self.__x = LpVariable.dicts("route", (range(depot), range(depot), range(self.__m)), cat="Binary")
        node_vars = LpVariable.dicts("node", (range(num_cities), range(self.__m)), lowBound=0, upBound=depot - 1, cat="Integer")
        max_route_distance = LpVariable("max_route_distance", lowBound=lower_bound, upBound=upper_bound, cat="Integer")


        courier_weights =[]
        self.__courier_dist=[]

        for courier in range(self.__m):

            weights = LpVariable(f"courier_weight_{courier}", lowBound=0, upBound=self.__L[courier], cat="Integer")
            courier_weights.append(weights)

            distance = LpVariable(f"courier_distance_{courier}", cat="Integer", lowBound=min_courier_dist, upBound=upper_bound)
            self.__courier_dist.append(distance)
        

        self.__lp_prob += max_route_distance

        # Weight constraints for each courier
        for courier in range(self.__m):
            self.__lp_prob += courier_weights[courier] == LpAffineExpression(
                [(self.__x[i][j][courier], self.__S[j]) for i in range(self.__n + 1) for j in range(self.__n)]
            )

        # Prevent self-looping arcs
        self.__lp_prob += lpSum(self.__x[i][i][courier] for i in range(depot) for courier in range(self.__m)) == 0

        # Each city visited exactly once
        for city in range(num_cities):
            self.__lp_prob += lpSum(self.__x[i][city][courier] for i in range(depot) for courier in range(self.__m)) == 1

        # Each courier departs from the depot
        for courier in range(self.__m):
            self.__lp_prob += lpSum(self.__x[num_cities][j][courier] for j in range(num_cities)) == 1

        # Each courier returns to the depot
        for courier in range(self.__m):
            self.__lp_prob += lpSum(self.__x[i][num_cities][courier] for i in range(num_cities)) == 1

        # Path connectivity constraints
        for city in range(depot):
            for courier in range(self.__m):
                self.__lp_prob += lpSum(self.__x[i][city][courier] for i in range(depot)) == lpSum(self.__x[city][i][courier] for i in range(depot))

        # Ensure no double usage of arcs
        for courier in range(self.__m):
            for i in range(num_cities):
                for j in range(num_cities):
                    self.__lp_prob += self.__x[i][j][courier] + self.__x[j][i][courier] <= 1

        # Subtour elimination constraints
        for courier in range(self.__m):
            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        self.__lp_prob += node_vars[i][courier] - node_vars[j][courier] + num_cities * self.__x[i][j][courier] <= num_cities - 1

        # Calculate distances for each courier
        for courier in range(self.__m):
            self.__lp_prob += lpSum(self.__x[i][j][courier] * self.__D[i][j] for i in range(depot) for j in range(depot)) == self.__courier_dist[courier]

        # Max distance constraint
        for dist in self.__courier_dist:
            self.__lp_prob += max_route_distance >= dist


    
    def __create_json(self, status, print_summary):
        """
        Writes the results of the solver into a JSON file in the required format.

        Args:
            status: Status of the solver (e.g., 1 if successful, 0 otherwise)
        """
        # Build the file path for the output JSON file
        json_file_path = os.path.join(self.__res_path, f"{self.__instance}.json")

        # Determine if the solution is optimal
        solve_time = 300 if floor(self.__lp_prob.solutionTime) > 300 else floor(self.__lp_prob.solutionTime)
        optimal = status == 1 and solve_time < self.__time_limit

        # Extract the solution if it exists
        solution = []
        for courier in range(self.__m):
            courier_route = []
            for i in range(self.__n + 1):
                for j in range(self.__n):
                    if value(self.__x[i][j][courier]) > 0.5:  # Decision variable is binary
                        courier_route.append(j + 1)  # Add 1 for 1-based indexing
            solution.append(courier_route)

        # Prepare the result entry for this solver
        result_entry = {
            "CBC": {
                "time": solve_time,
                "optimal": optimal,
                "obj": value(self.__lp_prob.objective),
                "sol": solution,
            }
        }

        print(result_entry)

        if print_summary:
            # If the JSON file already exists, load it and append the new entry
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Add the result entry to the JSON data
            data.update(result_entry)
            
            # Write the updated data back to the JSON file
            with open(json_file_path, "w") as file:
                json.dump(data, file, indent=4)
            
        return result_entry


    def run_model_on_instance(self, instance, print_summary=False):

        """
        Public method of the class. Reads the variables of the instance and starts the model
        """

        self.__instance=instance

        #building the file path from the instance number
        if self.__instance < 10:
            instances_path = self.__inst_path + "/inst0" + str(self.__instance) + ".dat"  #create the filepath
        else:
            instances_path = self.__inst_path + "/inst" + str(self.__instance) + ".dat"  #create the filepath


        #initialize all the problem variables
        self.__m, self.__n, self.__L,  self.__S, self.__D =  read_dat_file(instances_path)
            

        self.__configure_problem()
        highs=getSolver('PULP_CBC_CMD', timeLimit=self.__time_limit,msg=False)
        self.__lp_prob.solve(highs)

        status=self.__lp_prob.status
        #print(f'++++++++++++++++++++++ \n{status}')
        if status == 1:
            solve_time= 300 if floor(self.__lp_prob.solutionTime) > 300 else floor(self.__lp_prob.solutionTime)
            opt=(self.__time_limit > solve_time)
            #print(f'++++++SolveTime:{self.__lp_prob.objective, solve_time}++++++')
            #print(f'++++++++++++++++++++ \n OBJECTIVE VALUE: {value(self.__lp_prob.objective)} \n++++++++++++++++++++')
        else:
            
            print('Failed to find a solution.....')

        
        results =self.__create_json(status, print_summary)
        return results
            
    def get_x(self):
        return self.__x
    

#function which takes as an input an array with the instances to solve and the path to the instances

def run_instances(instances, inst_path, res_path, print_summary, time_limit=300,):
    """
    Run the MIP solver on a list of instances and write the results to a JSON file.

    Args:
        instances (list): List of instance numbers to solve
        inst_path (str): Path to the folder containing the instances
        res_path (str): Path to the folder to write the results
        time_limit (int): Time limit for the solver in seconds
    """
    # Initialize the MIP solver

    # Solve each instance
    for instance in instances:
        solver = MIP_solver(res_path, inst_path, time_limit)
        print(f"Solving instance {instance}...")
        solver.run_model_on_instance(instance, print_summary)

    print("All instances solved.")

if __name__ == "__main__":
    # Define and parse the arguments
    parser = argparse.ArgumentParser(description="A MIP solver script with customizable arguments.")
    parser.add_argument("--instances", "-i", required=True, help="Path to the folder containing instance files.")
    parser.add_argument("--results", "-r", required=True, help="Path to the base folder where results will be stored.")
    parser.add_argument("--folder-name", "-f", default="MIP", 
                        help="Name of the subfolder to store results (default: MIP).")
    parser.add_argument("--selected", "-s", default="1,2,3,4,5,6,7,8,9,10", 
                        help="An integer or a list of integers specifying selected instances "
                             "(default: \"1,2,3,4,5,6,7,8,9,10\").")
    parser.add_argument("--time-limit", "-t", type=int, default=300, 
                        help="Time limit for solving each instance in seconds (default: 300).")
    parser.add_argument("--print-summary", "-p", action="store_true",
                    help="Enable verbose output (default: False)")

    args = parser.parse_args()

    # Parse and handle the selected_instances argument
    if args.selected.isdigit():
        selected_instances = [int(args.selected)]
    else:
        selected_instances = list(map(int, args.selected.strip("[]").split(",")))

    print("MIP solver started")

    # Paths
    inst_path = os.path.abspath(args.instances)
    res_path = os.path.join(os.path.abspath(args.results), args.folder_name)

    # Create the results folder if it doesn't exist
    os.makedirs(res_path, exist_ok=True)

    print(f"Print summary: {args.print_summary}")
    print(f"Selected instances: {selected_instances}")
    print(f"Results will be stored in: {res_path}")
    print(f"Time limit per instance: {args.time_limit} seconds")

    # Run the MIP solver on the selected instances
    run_instances(selected_instances, inst_path, res_path, args.print_summary, args.time_limit)

    print("MIP solver done")





#python3 mip.py -i './instances/' -r './../res/MIP' -s '11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21' --print-summary 'True'