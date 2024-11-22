%TO TRY:

% build a HEURISTIC for couriers. Also think about search Heuristic (variable/values assignment like first fail etc.) Maybe it can be smart to set the "assign" decision variable to an even number of package to everyone?

% try to look for global constraint

% beware of redundant constraints! they are computationally expensive. symmetry breaking may make some constraints redundant. BUT ALSO, they may exploit new implied constraint!

% set a maximum value for the optimization function and print all the solutions to see how the three is explored. In this way you can understand the search strategy.

% implement REVERSE_PATH with channeling constraint and a second_path that contains the path as in the first model. there you can exploit some symmetry constraints maybe!

% build a global path from the path matrix that keeps track of a global path. e.g. depot->1->depot->2->3->depoth. i dunno if this works with subcircuit constraint: if not can avoid to write the depot in the path.

% find annotation like ::domain to improve the search for the solve function

include "globals.mzn";
% TRY TO PUT THEM AS VARIABLES
int: m; % Number of couriers
int: n; % Number of packages
array [1..m] of var int: max_load; % Maximum load capacity for each courier
array [1..n] of var int: weights;  % Weights for each package
array[1..n + 1, 1..n + 1] of var int: distances;  % Distance matrix (n+1)x(n+1)
array [1..n] of var 1..m: assign; % Assignment array: each package assigned to a courier
array[1..m, 1..n + 1] of var 1..(n+1): path;  % Path matrix for each courier
array[1..m,1..n] of var 0..1: packages; % packages[c,i] = 1 means packages i is delivered by c.
array[1..m] of var int: actual_load;
% array[1..m] of var 1..n: item_delivered_by; % counting the number of items delivered by each courier. in this way i can also say that every courier must deliver at least one item!
% var int: min_distance_value;
int: max_distance_value = 1500;

% Constraints
% constraint min_distance_value = min(i in 1..n)(distances[i,n+1]+distances[n+1,i]);
% constraint max_distance_value = sum(i in 1..n+1)(max(j in 1..n+1)(distances[i,j]));

% Load constraint
% constraint bin_packing_capa(max_load, assign, weights); % global version
constraint forall(c in 1..m)(sum(i in 1..n where path[c,i]!=i)(weights[i]) <= max_load[c]);
    
% To fill path already with non visitable location by the respective courier
constraint forall(c in 1..m)  (
    forall(i in 1..n) (
    (assign[i] != c <-> path[c, i] = i) /\
    (assign[i] = c <-> path[c, i] != i) 
    )
    /\    path[c, n+1] != n+1 
);

% % counting items delivered by each courier (EXPENSIVE TO COMPUTE, NOT REALLY NEEDED) (for large instances using gecode)
% constraint forall(c in 1..m) ( 
%       item_delivered_by[c] = count(assign, c) /\ 
%       sum(p in 1..n)(packages[c,p]) = item_delivered_by[c]
% );

% channelling between assign and packages
constraint forall(c in 1..m)(
    forall(i in 1..n)(
      (assign[i] = c <-> packages[c,i] = 1) /\
      (assign[i] != c <-> packages[c,i] = 0)
    )
);

% if every courier sends only a package all the columns of packages must be different
constraint alldifferent([assign[i]|i in 1..n]) -> forall (p in 1..n)(
    alldifferent([packages[c,p]|c in 1..m])
);

% every courier must visit a different point as first one. (implied constraint)
constraint alldifferent([path[c,n+1]|c in 1..m]); %HELP A LOT

% implied constraint with packages. every packages must be shipped by only once
constraint forall(p in 1..n)(
    sum(c in 1..m)(packages[c,p]) = 1
);

% counting the load
constraint forall(c in 1..m)(
    actual_load[c] = sum(p in 1..n where packages[c,p] = 1)(weights[p]) 
);


% Constraint symmetry breaking: % seems to be helpful. 4x faster on instance2
% constraint forall(c1,c2 in 1..m where c1<c2) (
%     (max_load[c1] = max_load[c2] \/ ((actual_load[c1] <= max_load[c2])/\ (actual_load[c2] <= max_load[c1]))) 
%     ->(
%     lex_lesseq([assign[p] = c1 | p in 1..n], [assign[p] = c2 | p in 1..n])
% %     /\ lex_lesseq([path[c1,i]|i in 1..n+1],[path[c2,i]|i in 1..n+1]) %should be the same as the one above
%     /\ lex_lesseq([packages[c1,i]|i in 1..n],[packages[c2,i]|i in 1..n]) % THIS SEEMS TO IMPACT A BIT BUT IM NOT SURE WHETHER POSITIVELY OR NEGATIVELY
    
%     )
% );

%HEURISTIC: 2 COURIERS THAT HAVEN'T FILLED THEIR CAPACITY CAN BE EXCHANGED
% constraint forall(c1,c2 in 1..m where c1<c2) (
%     ((actual_load[c1] <= max_load[c2])/\ (actual_load[c2] <= max_load[c1])) ->(
%     lex_lesseq([assign[p] = c1 | p in 1..n], [assign[p] = c2 | p in 1..n])
% %     /\ lex_lesseq([path[c1,i]|i in 1..n+1],[path[c2,i]|i in 1..n+1]) %should be the same as the one above
%      /\ lex_lesseq([packages[c1,i]|i in 1..n],[packages[c2,i]|i in 1..n]) % THIS SEEMS TO IMPACT A BIT BUT IM NOT SURE WHETHER POSITIVELY OR NEGATIVELY
%     )
% );


% to say that a point cannot be visited multiple times
constraint forall(c in 1..m) (
    % Constrain path to contain different items
%       alldifferent([path[c,i] | i in 1..n+1]) /\
      subcircuit([path[c,i] | i in 1..n+1])%constraint subcircuit 
      
);


% Constraint: Each package is visited only by its assigned courier and exactly once
constraint forall(p in 1..n) ( % experimentally works even without this. IT IS EITHER REDUNDANT OR IMPLIED
    let {
        var int: assigned_courier = assign[p]
    } in
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
                distances[i,path[c,i]]
        )
);
% I MANAGED TO COMPUTE THE DISTANCE BUT IM NOT SURE IT'S OK. THO IT SHOULD BE OK

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the maximum traveled distance among all couriers
var 1..max_distance_value: max_distance;  % Maximum distance traveled by any courier


% Constraint to link the max distance to the total distances
constraint max_distance = max(c in 1..m) (total_distance[c]);

% Objective: Minimize the maximum distance traveled by any courier
solve 
%   :: seq_search([
%   int_search(packages,dom_w_deg, indomain_random, complete),
%   int_search(assign,dom_w_deg, indomain_min, complete),
%   int_search(item_delivered_by, first_fail, indomain_min), % 1st: dom_w_deg for the "assignment" variable
%   int_search(path, first_fail, indomain_min),        % 2nd: first_fail for the "path" variable  
% ])
% ::restart_luby(10)
minimize max_distance;
% satisfy

% Specify the LNS strategy with 95% retain
% Assuming an array of decision variables `all_vars` representing your full set of variables
% solve minimize max_distance;
% solve satisfy;
% solve minimize max_distance;
% solve :: int_search(assign, first_fail, indomain_min, complete) minimize max_distance;
