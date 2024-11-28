from z3 import *
import time
from utils.utils_jana import process_instances_input, current_datetime

def round_trip_distance(item, depot, distance_matrix):
    return distance_matrix[depot][item] + distance_matrix[item][depot]

def extract_ordered_route(model, courier_id, num_items,x, y, d,item_encodings):
    print('TUKS')
    ordered_route = []
    assigned_items = [j for j in range(num_items) if is_true(model.evaluate(x[courier_id][j]))]
    # TODO: print([j for j in range(num_items) if is_true(model.evaluate(y[courier_id][num_items][j]))])
    for p1 in range(num_items+1):
        for q1 in range(num_items+1):
            if model.evaluate(y[courier_id][p1][q1]):
                print(f"p1 {p1} q1 {q1} {model.evaluate(y[courier_id][p1][q1])}")
    print(f" Distance {model.evaluate(d[courier_id])}")
    current_location = num_items
    while True:
        for q in assigned_items + [num_items]:
            #print(f"p {current_location} q {q}")
            #print(model.evaluate(y[courier_id][current_location][q]))
            if current_location!=q and is_true(model.evaluate(y[courier_id][current_location][q])):
                ordered_route.append(q)
                print(ordered_route)
                if len(ordered_route) == len(assigned_items):
                    #return [k+1 for k in ordered_route]
                    print(f"Ordered {[k+1 for k in ordered_route]}")
                    print(assigned_items)
                    print('VRANJAM')
                    return [item_encodings.get(k+1) for k in ordered_route]
                current_location = q
                break
    return

def check_elapsed_time(start_time,timeout,print_time=True):
    elapsed_time = time.time() - start_time
    if print_time:
        print(f"elapsed time: {elapsed_time}")
    return elapsed_time > timeout

def compute_larger_bound(matrix):
    from heapq import nlargest

    n = len(matrix)
    used_columns = set()
    total_sum = 0

    for row in matrix:
        # Find the largest valid (unused column) weights in this row
        best_values = [(value, col) for col, value in enumerate(row) if col not in used_columns]
        if best_values:
            largest_value, col = max(best_values)
            total_sum += largest_value
            used_columns.add(col)

    return total_sum


# Compute the lower bound as the max of the round-trip distances for all items
def compute_lower_bound(depot, num_items, distance_matrix):
    max_distance = 0
    for item in range(num_items):
        # Calculate the round-trip distance for each item
        round_trip_dist = round_trip_distance(item, depot, distance_matrix)
        max_distance = max(max_distance, round_trip_dist)

    return max_distance

def solve_mcp_z3(num_couriers, num_items, load_limits, item_sizes, int_distance_matrix, verbose=False):
    # Verbose logging function
    def log(message):
        if verbose:
            current_datetime()
            print(f"[LOG] {message}")

    # Initialize the Z3 optimizer
    opt = Optimize()

    # Set a timeout (in milliseconds)
    timeout = 300  # 300 seconds

    log("Initialization complete. Setting up variables and constraints...")

    start_time = time.time()
    # Sort the items
    sorted_indices_item = sorted(range(len(item_sizes)), key=lambda k: item_sizes[k])
    item_sizes = [item_sizes[i] for i in sorted_indices_item]
    sorted_indices_item = sorted_indices_item + [len(item_sizes)]
    item_encodings = dict(zip(list(range(1,num_items+1)),[inx +1 for inx in sorted_indices_item]))

    # Sort the distance matrix rows and columns according to the sorted items
    int_distance_matrix = [[int_distance_matrix[i][j] for j in sorted_indices_item] for i in sorted_indices_item]

    # Sort couriers by capacity (ascending)
    sorted_indices_load = sorted(range(len(load_limits)), key=lambda k: load_limits[k])
    load_limits = [load_limits[i] for i in sorted_indices_load]

    distance_matrix = [[IntVal(int_distance_matrix[p][q]) for q in range(num_items + 1)]
                       for p in range(num_items + 1)]

    # Decision Variables:
    # x[i][j] = 1 if courier i is assigned to item j, 0 otherwise
    x = [[Bool(f'x_{i}_{j}') for j in range(num_items)] for i in range(num_couriers)]

    # y[i][p][q] is now a bitvector instead of a bool matrix
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
    log(f"Bounds set. Upper: {upper_bound}, Lower: {lower_bound}")

    # Breaking courier symmetry - Enforce order by distance traveled
    for i in range(num_couriers - 1):
        opt.add(d[i] <= d[i + 1])

    if check_elapsed_time(start_time, timeout):
        print('1 Elapsed time too much')
        return

    # Constraints:
    log("Adding constraints...")

    # Each item must be assigned to exactly one courier DEMAND FULFILLMENT
    for j in range(num_items):
        opt.add(Sum([If(x[i][j], 1, 0) for i in range(num_couriers)]) == 1)

    if check_elapsed_time(start_time, timeout):
        print('2 Elapsed time too much')
        return

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

    if check_elapsed_time(start_time, timeout):
        print('3 Elapsed time too much')
        return

    # Early Exclusion of Unusable Couriers
    for i in range(num_couriers):
        for j in range(num_items):
            if item_sizes[j] > load_limits[i]:
                opt.add(Not(x[i][j]))
                for k in range(j + 1, num_items):
                    opt.add(Not(x[i][k]))

    if check_elapsed_time(start_time, timeout):
        print('4 Elapsed time too much')
        return

    # Ensure each courier has at least one item
    for i in range(num_couriers):
        opt.add(Sum([x[i][j] for j in range(num_items)]) >= 1)

    if check_elapsed_time(start_time, timeout):
        print('5 Elapsed time too much')
        return

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

    # # Ensure no revisits to the same location for each courier
    # for i in range(num_couriers):
    #     for p in range(num_items):
    #         opt.add(And([
    #             Implies(And(y[i][p][q] == 1, y[i][q][r] == 1), y[i][r][p] == 0)
    #             for q in range(num_items) for r in range(num_items)
    #             if q != p and r != p and q != r
    #         ]))

    if check_elapsed_time(start_time, timeout):
        print('6 Elapsed time too much')
        return

    # # Route Continuity: Only include y[i][p][q] if x[i][q] is true (i.e., courier i is assigned to item q)
    # for i in range(num_couriers):
    #     for p in range(num_items + 1):  # Including depot
    #         for q in range(num_items):  # Exclude depot as it's only the starting/ending point
    #             opt.add(Implies(y[i][p][q] == 1, x[i][q]))


    # Preventing redundant "revisits" to a location by ensuring only allowed transitions
    for i in range(num_couriers):
        for j in range(num_items):
            opt.add(Sum([If(y[i][p][j], 1, 0) for p in range(num_items + 1)]) == If(x[i][j], 1, 0))
            opt.add(Sum([If(y[i][j][q], 1, 0) for q in range(num_items + 1)]) == If(x[i][j], 1, 0))
        opt.add(Not(y[i][num_items][num_items]))
        opt.add(Sum([If(y[i][p][num_items], 1, 0) for p in range(num_items + 1)]) == 1)
        opt.add(Sum([If(y[i][num_items][q], 1, 0) for q in range(num_items + 1)]) == 1)

    if check_elapsed_time(start_time, timeout):
        print('7 Elapsed time too much')
        return

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
    log("Starting solver...")
    time_left = max(1,timeout - int(time.time() - start_time))
    opt.set("timeout", time_left * 1000)
    log(f"Timeout: {time_left}")
    current_datetime()
    log('cico <3')
    # Solve the problem with a timer
    result = opt.check()
    runtime = time.time() - start_time

    optimal = False
    obj = None
    routes = []
    ended = ""

    if result == sat or result == unknown:
        model = opt.model()
        # If the result is sat, we assume it's optimal
        if result == sat:
            optimal = True
            ended = "SAT"
        else:
            optimal = False  # Unknown means timeout reached without proving optimality
            ended = "Timeout"

        # Extract the objective value
        if model.eval(max_distance, model_completion=True) is not None:
            obj = model[max_distance].as_long()

        # Extract routes for each courier
        for i in range(num_couriers):
            # assigned_items = [j + 1 for j in range(num_items) if is_true(model.evaluate(x[i][j]))]
            # routes.append(assigned_items)
            route = extract_ordered_route(model, i, num_items,x, y, d, item_encodings)
            routes.append(route)
            log(f'Courier {i + 1}: Assigned items: {route}, Distance: {model[d[i]]}')
    else:
        log("No solution found.")
        ended = "No solution exists"

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
    print('SAT solver started')
    inst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[1])
    os.makedirs(args[2], exist_ok=True)
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args[2])
    process_instances_input(inst_path,res_path,solve_mcp_z3,"try_without_optimal",verbose=True)
    print('SAT solver done')