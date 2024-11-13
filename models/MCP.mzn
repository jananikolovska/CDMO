include "globals.mzn";

% Use this editor as a MiniZinc scratch book
int: m; % Number of couriers
int: n; % Number of packages
array [1..m] of int: max_load; % Maximum load capacity for each courier
array [1..n] of int: weights;  % Weights for each package
array[1..n + 1, 1..n + 1] of int: distances;  % Distance matrix (n+1)x(n+1)
int: max_distance_value = 10000;
array [1..n] of var 1..m: assign; % Assignment array: each package assigned to a courier
array[1..m, 1..n + 1] of var 1..(n+1): path;  % Path matrix for each courier
array[1..m] of var 1..n: item_delivered_by; % counting the number of items delivered by each courier. in this way i can also say that every courier must deliver at least one item!


constraint forall(c in 1..m) (
    item_delivered_by[c] = sum([assign[p] = c | p in 1..n]) % counting items delivered by each courier
);


% Load constraint: The total weight of packages carried by each courier should not exceed their max load
% constraint forall(c in 1..m) (
%     sum(p in 1..n where assign[p] = c) (weights[p]) <= max_load[c]
% );
constraint bin_packing_capa(max_load, assign, weights); % global version

% Constraints
% Each courier starts and ends their route at the dispatch center
constraint forall(c in 1..m) (
    path[c, 1] = n + 1 /\
    forall(i in item_delivered_by[c] + 2..n + 1 where i <= n + 1) ( % we are padding the path variable with dispatch location.
      path[c, i] = n+1         
    )
);

% to say 
constraint forall(c in 1..m) (
    let {
        set of int: s = {n+1}
    } in
    % Constrain non-padded part of path to contain different items
      alldifferent_except([path[c,i] | i in 2..n], s)
);

% Constraint: Each package is visited only by its assigned courier and exactly once
constraint forall(p in 1..n) (
    let {
        var int: assigned_courier = assign[p]
    } in
        % Ensure that the package is visited by the assigned courier
        exists(pos in 2..n) (  % start from pos 2 to leave space for dispatch center at the start
            path[assigned_courier, pos] = p
        ) /\
        % Ensure that no other courier visits this package
        forall(c in 1..m where c != assigned_courier) (
            forall(pos in 2..n) (
                path[c, pos] != p
            )
        )
);

% Calculate the total distance for each courier
array[1..m] of var 0..max_distance_value: total_distance;  % Total distance traveled by each courier

constraint forall(c in 1..m) (
    total_distance[c] = 
        sum(pos in 1..(n)) (
            distances[path[c, pos], path[c, pos + 1]]
        )
);

% Define the maximum traveled distance among all couriers
var 0..max_distance_value: max_distance;  % Maximum distance traveled by any courier

% Constraint to link the max distance to the total distances
constraint max_distance = max(c in 1..m) (total_distance[c]);

% Objective: Minimize the maximum distance traveled by any courier
solve minimize max_distance;
% solve satisfy;
