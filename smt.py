from z3 import *
from utils.utils import *

def round_trip_distance(item, depot, distance_matrix):
    return distance_matrix[depot][item] + distance_matrix[item][depot]

def extract_ordered_route(model, courier_id, num_items,x, y, item_encodings):
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
    max_distance = 0
    for item in range(num_items):
        # Calculate the round-trip distance for each item
        round_trip_dist = round_trip_distance(item, depot, distance_matrix)
        max_distance = max(max_distance, round_trip_dist)

    return max_distance

def solve_mcp_z3(num_couriers, num_items, load_limits, item_sizes, int_distance_matrix, verbose=True):
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
    # x[i][j] = 1 if courier i is assigned to item j, 0 otherwise
    x = [[Bool(f'x_{i}_{j}') for j in range(num_items)] for i in range(num_couriers)]

    # y[i][p][q] = 1 if courier i travels from point p to point q, 0 otherwise
    y = [[[Bool(f'y_{i}_{p}_{q}') for q in range(num_items + 1)]
          for p in range(num_items + 1)] for i in range(num_couriers)]

    order = [[Int(f"order_{i}_{j}") for j in range(num_items)] for i in range(num_couriers)]

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
    log(f"Bounds set. Upper: {upper_bound}, Lower: {lower_bound}", verbose)

    # Breaking courier symmetry - Enforce order by distance traveled
    for i in range(num_couriers - 1):
        opt.add(d[i] <= d[i + 1])

    check_elapsed_time(start_time, timeout)

    # Constraints:
    log("Adding constraints...",verbose)

    # Each item must be assigned to exactly one courier DEMAND FULFILLMENT
    for j in range(num_items):
        opt.add(Sum([If(x[i][j], 1, 0) for i in range(num_couriers)]) == 1)

    check_elapsed_time(start_time, timeout)

    # Iterate over couriers and items
    for i in range(num_couriers):
        for k in range(num_items):
            # Create the sum of x[i][j] over all j items
            sum_x = Sum([If(x[i][j], 1, 0) for j in range(num_items)])

            # Check if sum_x > 1, then add the condition that you can't have both transitions
            opt.add(Implies(sum_x > 1,
                            Not(And(y[i][num_items][k], y[i][k][num_items]))))

    # Each courier's load must not exceed their capacity CAPACITY CONSTRAINT
    for i in range(num_couriers):
        # Add the load constraint for each courier
        opt.add(Sum([If(x[i][j], item_sizes[j], 0) for j in range(num_items)]) <= load_limits[i])

    check_elapsed_time(start_time, timeout)

    # Early Exclusion of Unusable Couriers
    for i in range(num_couriers):
        for j in range(num_items):
            if item_sizes[j] > load_limits[i]:
                opt.add(Not(x[i][j]))
                for k in range(j + 1, num_items):
                    opt.add(Not(x[i][k]))

    check_elapsed_time(start_time, timeout)

    # Ensure each courier has at least one item
    for i in range(num_couriers):
        opt.add(Sum([x[i][j] for j in range(num_items)]) >= 1)

    check_elapsed_time(start_time, timeout)

    #Route Continuity
    for i in range(num_couriers):
        for p in range(num_items):
            opt.add(Not(y[i][p][p]))  # Ensure no self-loop (y[i][p][p] = 0)
            for q in range(p + 1, num_items):  # Only consider pairs where p < q to avoid duplication
                opt.add(Implies(y[i][p][q], Not(y[i][q][p])))

    for i in range(num_couriers):
        for p in range(num_items):
            opt.add(Implies(Not(x[i][p]), order[i][p] < 0))
            opt.add(Implies(x[i][p], order[i][p] > 0))
            for q in range(num_items):
                # If there is a transition from p to q, enforce order[p] < order[q]
                opt.add(Implies(y[i][p][q], order[i][p] < order[i][q]))

    for i in range(num_couriers):
        opt.add(Distinct([order[i][p] for p in range(num_items)]))

    check_elapsed_time(start_time, timeout)

    # Preventing redundant "revisits" to a location by ensuring only allowed transitions
    for i in range(num_couriers):
        for j in range(num_items):
            opt.add(Sum([If(y[i][p][j], 1, 0) for p in range(num_items + 1)]) == If(x[i][j], 1, 0))
            opt.add(Sum([If(y[i][j][q], 1, 0) for q in range(num_items + 1)]) == If(x[i][j], 1, 0))
        opt.add(Not(y[i][num_items][num_items]))
        opt.add(Sum([If(y[i][p][num_items], 1, 0) for p in range(num_items + 1)]) == 1)
        opt.add(Sum([If(y[i][num_items][q], 1, 0) for q in range(num_items + 1)]) == 1)

    check_elapsed_time(start_time, timeout)

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
    log("Constraints added. Starting solver...",verbose)

    # Set a timeout (in milliseconds)
    time_left = max(1,timeout - int(time.time() - start_time))
    opt.set("timeout", time_left * 1000)
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

    return result

if __name__ == "__main__":
    args = sys.argv
    log('SMT solver started')
    inst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[1])
    os.makedirs(args[2], exist_ok=True)
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[2])
    process_instances_input(inst_path,res_path,solve_mcp_z3,"SMT")
    print('SAT solver done')