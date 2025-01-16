from z3 import *
from utils.utils import *

def round_trip_distance(item, depot, distance_matrix):
    """
        Computes the round-trip distance for an item from the depot and back.

        Args:
            item (int): The index of the item.
            depot (int): The index of the depot.
            distance_matrix (list of list of ints): A 2D matrix representing the distances between points.

        Returns:
            int: The sum of the distance from depot to item and back.
    """
    return distance_matrix[depot][item] + distance_matrix[item][depot]

def extract_ordered_route(model, courier_id, num_items,x, y, item_encodings):
    """
        Extracts the ordered route for a specific courier from the Z3 model.

        Args:
            model (z3.Model): The Z3 model containing the decision variables.
            courier_id (int): The index of the courier.
            num_items (int): The total number of items.
            x (list of list of z3.Bool): The assignment matrix for items to couriers.
            y (list of list of list of z3.Bool): The travel path matrix for couriers.
            item_encodings (dict): A dictionary mapping item indices to encodings.

        Returns:
            list: A list of item encodings in the order they are assigned to the courier.
    """
    ordered_route = []
    assigned_items = [j for j in range(num_items) if is_true(model.evaluate(x[courier_id][j]))]
    current_location = num_items
    while True:
        for q in assigned_items + [num_items]:
            if current_location!=q and is_true(model.evaluate(y[courier_id][current_location][q])):
                ordered_route.append(q)
                print(ordered_route)
                if len(ordered_route) == len(assigned_items):
                    return [item_encodings.get(k+1) for k in ordered_route]
                current_location = q
                break
    return

def compute_larger_bound(matrix):
    """
       Computes a larger bound for the optimization problem by selecting the maximum values from the matrix.

       Args:
           matrix (list of list of ints): A 2D matrix representing the distances or costs.

       Returns:
           int: The sum of the largest values from each row of the matrix.
    """
    used_columns = set()
    total_sum = 0

    for row in matrix:
        best_values = [(value, col) for col, value in enumerate(row) if col not in used_columns]
        if best_values:
            largest_value, col = max(best_values)
            total_sum += largest_value
            used_columns.add(col)

    return total_sum

def compute_lower_bound(depot, num_items, distance_matrix):
    """
        Computes a lower bound for the optimization problem based on the round-trip distances from the depot.

        Args:
            depot (int): The index of the depot.
            num_items (int): The total number of items.
            distance_matrix (list of list of ints): A 2D matrix representing the distances between points.

        Returns:
            int: The maximum round-trip distance for any item from the depot.
    """
    max_distance = 0
    for item in range(num_items):
        # Calculate the round-trip distance for each item
        round_trip_dist = round_trip_distance(item, depot, distance_matrix)
        max_distance = max(max_distance, round_trip_dist)

    return max_distance

def create_variables(num_couriers, num_items):
    x = [[Bool(f'x_{i}_{j}') for j in range(num_items)] for i in range(num_couriers)]
    y = [[[Bool(f'y_{i}_{p}_{q}') for q in range(num_items + 1)] for p in range(num_items + 1)] for i in range(num_couriers)]
    order = [[Int(f"order_{i}_{j}") for j in range(num_items)] for i in range(num_couriers)]
    d = [Int(f'd_{i}') for i in range(num_couriers)]
    return x, y, order, d

def set_bounds(int_distance_matrix, num_items, opt, max_distance):
    upper_bound = compute_larger_bound(int_distance_matrix)
    lower_bound = compute_lower_bound(num_items, num_items, int_distance_matrix)
    opt.add(max_distance <= upper_bound)
    opt.add(max_distance >= lower_bound)
    return upper_bound, lower_bound

def solve_mcp_z3(num_couriers,
                 num_items,
                 load_limits,
                 item_sizes,
                 int_distance_matrix,
                 res_path,
                 inst_id,
                 solver_name,
                 verbose=True):
    """
        Solves the Multi-Courier Problem (MCP) using the Z3 solver.

        Args:
            num_couriers (int): The number of couriers.
            num_items (int): The number of items to be delivered.
            load_limits (list of ints): The load limits for each courier.
            item_sizes (list of ints): The sizes of each item.
            int_distance_matrix (list of list of ints): The distance matrix between items and depot.
            verbose (bool): Flag to enable detailed logging.

        Returns:
            dict: A dictionary containing the execution time, optimal solution status, objective value, and the solution routes.
    """
    start_time = time.time()
    timeout = 300
    current_datetime()

    # Initialize the Z3 optimizer
    opt = Optimize()
    log("Initialization complete. Setting up variables and constraints...", verbose)

    item_sizes, load_limits , int_distance_matrix, (item_encodings,sorted_indices_load) = encode_input(item_sizes, load_limits, int_distance_matrix,
                                                                                 sort_items=True, sort_loads=True)

    distance_matrix = [[IntVal(int_distance_matrix[p][q]) for q in range(num_items + 1)]
                       for p in range(num_items + 1)]

    # Decision Variables:
    x, y, order, d = create_variables(num_couriers, num_items)

    # max_distance = the maximum distance traveled by any courier
    max_distance = Int('max_distance')

    # Initialize lower bound
    upper_bound, lower_bound = set_bounds(int_distance_matrix, num_items, opt, max_distance)
    log(f"Bounds set. Upper: {upper_bound}, Lower: {lower_bound}", True)

    log("Adding constraints...", verbose)
    # Breaking courier symmetry - Enforce order by distance traveled
    for i in range(num_couriers - 1):
        opt.add(d[i] <= d[i + 1])

    # Constraints:

    # DEMAND FULFILLMENT
    for j in range(num_items):
        opt.add(Sum([If(x[i][j], 1, 0) for i in range(num_couriers)]) == 1)

    # (optional) PREVENTING REDUNDANT TRANSITIONS FOR COURIERS WITH MULTIPLE ITEMS
    for i in range(num_couriers):
        for k in range(num_items):
            # how many items the courier i is assigned
            sum_x = Sum([If(x[i][j], 1, 0) for j in range(num_items)])

            # Check if sum_x > 1, then add the condition that you can't have both transitions (at least one more item before going back to depot)
            opt.add(Implies(sum_x > 1,
                            Not(And(y[i][num_items][k], y[i][k][num_items]))))

    # CAPACITY CONSTRAINT
    for i in range(num_couriers):
        # Add the load constraint for each courier
        opt.add(Sum([If(x[i][j], item_sizes[j], 0) for j in range(num_items)]) <= load_limits[i])

    # EARLY EXCLUSION OF UNDELIVERABLE ITEMS
    for i in range(num_couriers):
        for j in range(num_items):
            if item_sizes[j] > load_limits[i]:
                opt.add(Not(x[i][j]))
                for k in range(j + 1, num_items):
                    opt.add(Not(x[i][k]))

    # AT LEAST ONE ITEM PER COURIER
    for i in range(num_couriers):
        opt.add(Sum([x[i][j] for j in range(num_items)]) >= 1)

    #NO LOOP CONNECTION & DIRECTION OF ROUTE
    for i in range(num_couriers):
        for p in range(num_items):
            opt.add(Not(y[i][p][p]))  # Ensure no self-loop (y[i][p][p] = 0)
            for q in range(p + 1, num_items):  #Ensures no reverse transitions
                opt.add(Implies(y[i][p][q], Not(y[i][q][p])))

    #VARIABLE LINK: ASSIGNMENTS AND ORDER
    for i in range(num_couriers):
        for p in range(num_items):
            #connecting x variable with order variable
            opt.add(Implies(Not(x[i][p]), order[i][p] < 0))
            opt.add(Implies(x[i][p], order[i][p] > 0))
            for q in range(num_items):
                # If there is a transition from p to q, enforce order[p] < order[q]
                opt.add(Implies(y[i][p][q], order[i][p] < order[i][q]))

    #UNIQUENESS OF ORDER VARIABLE
    for i in range(num_couriers):
        opt.add(Distinct([order[i][p] for p in range(num_items)]))

    # VARIABLE LINK: ASSIGNMENTS AND ROUTE
    for i in range(num_couriers):
        for j in range(num_items):
            opt.add(Sum([If(y[i][p][j], 1, 0) for p in range(num_items + 1)]) == If(x[i][j], 1, 0))
            opt.add(Sum([If(y[i][j][q], 1, 0) for q in range(num_items + 1)]) == If(x[i][j], 1, 0))
        opt.add(Not(y[i][num_items][num_items]))
        opt.add(Sum([If(y[i][p][num_items], 1, 0) for p in range(num_items + 1)]) == 1)
        opt.add(Sum([If(y[i][num_items][q], 1, 0) for q in range(num_items + 1)]) == 1)

    #DISTANCE PER COURIER CALCULATION
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
    log("Constraints added. Starting solver...",verbose)

    # Set a timeout (in milliseconds)
    time_left = max(1,timeout - int(time.time() - start_time))
    log(f"Time left for solution: {time_left}",verbose)
    current_datetime()

    # Solve the problem with a timer
    result = opt.check()
    runtime = time.time() - start_time

    optimal = False
    obj = None
    routes = []

    if result == sat or result == unknown:
        model = opt.model()
        if result == sat:
            optimal = True
        else:
            optimal = False

        # Extract the objective value
        if model.eval(max_distance, model_completion=True) is not None:
            obj = model[max_distance].as_long()

        # Extract routes for each courier
        for i in range(num_couriers):
            route = extract_ordered_route(model, i, num_items,x, y, item_encodings)
            routes.append(route)
            log(f'Courier {i + 1}: Assigned items: {route}, Distance: {model[d[i]]}',verbose)
    else:
        log("No solution found.",verbose)

    exec_time = min(int(runtime), timeout)

    # Construct the JSON result
    result = {
        "time": exec_time,
        "optimal": optimal,
        "obj": obj if obj is not None else 0,  # Set to 0 if no objective found
        "sol":  [routes[i] for i in sorted_indices_load]
    }
    save_results(res_path, inst_id, solver_name, result)
    return result


if __name__ == "__main__":
    # Define and parse the arguments
    parser = argparse.ArgumentParser(description="A simple SMT solver script with customizable arguments.")
    parser.add_argument("--instances", "-i", required=True, help="Path to the folder containing instance files.")
    parser.add_argument("--results", "-r", required=True, help="Path to the base folder where results will be stored.")
    parser.add_argument("--folder-name", "-f", default="SMT",
                        help="Name of the subfolder to store results (default: SMT).")
    parser.add_argument("--selected", "-s", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21",
                        help="An integer or a list of integers specifying selected instances "
                             "(default: \"1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21\").")

    args = parser.parse_args()

    # Parse and handle the selected_instances argument
    if args.selected.isdigit():
        selected_instances = [int(args.selected)]
    else:
        selected_instances = list(map(int, args.selected.strip("[]").split(",")))

    log("SMT solver started")

    # Paths
    inst_path = os.path.abspath(args.instances)
    res_path = os.path.join(os.path.abspath(args.results), args.folder_name)

    # Create the results folder if it doesn't exist
    os.makedirs(res_path, exist_ok=True)

    log(f"Selected instances: {selected_instances}")
    # Process the instances
    process_instances_input(inst_path, res_path, selected_instances,args.folder_name, solve_mcp_z3, "SMT")

    log("SMT solver done")