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

% Constraints

% counting items delivered by each courier
constraint forall(c in 1..m) (
    item_delivered_by[c] = sum([assign[p] = c | p in 1..n]) 
);
% Load constraint
constraint bin_packing_capa(max_load, assign, weights); % global version

% To fill path already with non visitable location by the respective courier
constraint forall(c in 1..m)  (
    forall(i in 1..n) (
    (assign[i] != c -> path[c, i] = i) /\
    (assign[i] = c -> path[c, i] != i) 
    ) /\
    path[c, n+1] != n+1 
%     /\
%     forall (other_c in 1..m where c != other_c) (
%       forall(j in 1..n where j != i)(
%       path[other_c,j] != i
%       )
%     )
);


% to say that a point cannot be visited multiple times
constraint forall(c in 1..m) (
    % Constrain path to contain different items
      alldifferent([path[c,i] | i in 1..n+1]) /\
      subcircuit([path[c,i] | i in 1..n+1]) %constraint subcircuit
      
);

% Constraint: Each package is visited only by its assigned courier and exactly once
constraint forall(p in 1..n) (
    let {
        var int: assigned_courier = assign[p]
    } in
        % Ensure that the package is visited by the assigned courier
%         exists(pos in 1..n) (  % start from pos 2 to leave space for dispatch center at the start
%             path[assigned_courier, pos] != p 
%             )
        % Ensure that no other courier visits this package
        forall(c in 1..m where c != assigned_courier) (
            forall(pos in 1..n where pos != p) (
                path[c, pos] != p
            )
        )
);

% Calculate the total distance for each courier 
array[1..m] of var 0..max_distance_value: total_distance;  % Total distance traveled by each courier

constraint forall(c in 1..m) (
    total_distance[c] = 
        sum(i in 1..n+1) (
%             distances[path[c, pos], path[c, pos + 1]]
              if path[c,i]!= i then
                distances[i,path[c,i]]
              else 
                0
              endif
        )
);
% I MANAGED TO COMPUTE THE DISTANCE BUT IM NOT SURE IT'S OK. THO IT SHOULD BE OK

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the maximum traveled distance among all couriers
var 0..max_distance_value: max_distance;  % Maximum distance traveled by any courier

% Constraint to link the max distance to the total distances
constraint max_distance = max(c in 1..m) (total_distance[c]);

% Objective: Minimize the maximum distance traveled by any courier
solve minimize max_distance;
% solve satisfy;
