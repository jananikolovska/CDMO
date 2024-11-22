include "globals.mzn";

%       SETS
set of int: ITEMS = 1..n;
set of int: COURIERS = 1..m;
set of int: NODES = 1..n+1;

%       PARAMETERS
int: m; % Number of couriers
int: n; % Number of packages
array [COURIERS] of int: max_load; % Maximum load capacity for each courier
array [ITEMS] of int: weights;  % Weights for each package
array[NODES, NODES] of int: distances;  % Distance matrix (n+1)x(n+1)


%       DECISION VARIABLES
array [ITEMS] of var COURIERS: assign; % Assignment array: each package assigned to a courier
array[COURIERS, NODES] of var 1..(n+1): path;  % Path matrix for each courier

%        AUXILIARY VARIABLES
array[COURIERS,ITEMS] of var 0..1: packages; % packages[c,i] = 1 means packages i is delivered by c.
array[COURIERS] of var int: actual_load;
array [1..2] of var int: UB;
var int: LB; 
array[COURIERS] of var int: total_distance;

%        CONSTRAINTS

%        bounding the distance travelled by each courier
constraint LB = max(i in ITEMS)(distances[i,n+1]+distances[n+1,i]); % the maximum distance to carry a package.
constraint UB[1] = max(i in NODES)(sum(j in 1..n+1)(distances[i,j]));
constraint UB[2] = max(j in NODES)(sum(i in 1..n+1)(distances[i,j])); % the maximum distance to carry all the packages.
constraint forall (c in COURIERS)(LB <= max(total_distance[COURIERS]) /\  max(total_distance[COURIERS]) <= min(UB));

%        every row of the path variable must contain hamiltonian subcycles
constraint forall(c in COURIERS)(alldifferent(path[c,NODES]));
constraint forall(c in COURIERS)(subcircuit(path[c,NODES]));
constraint alldifferent(path[COURIERS,n+1]); % all the couriers must deliver a different packages as first one!
constraint forall(c in COURIERS)(path[c,n+1]!=n+1); % all the couriers must deliver at least one package!

%        the maximum loads must be respected for each courier
constraint forall(c in COURIERS)(sum(i in ITEMS where path[c,i]!=i)(weights[i]) <= max_load[c]);

%        CHANNELING constraints between path and assign
constraint forall(c in COURIERS)(forall(i in ITEMS)(
    (assign[i] != c <-> path[c, i] = i) 
    /\ (assign[i] = c <-> path[c, i] != i) 
    )
);

%        CHANNELING constraints between assign and packages.
constraint forall(c in COURIERS)(
    forall(i in ITEMS)(
      (assign[i] = c <-> packages[c,i] = 1) 
      /\(assign[i] != c <-> packages[c,i] = 0)
    )
);

%        CHANNELING constraints between path and packages
constraint forall(c in COURIERS)(
    forall(i in ITEMS)(
      (path[c,i] != i <-> packages[c,i] = 1) 
      /\(path[c,i] = i <-> packages[c,i] = 0)
    )
);

constraint forall(p in ITEMS)(sum(c in COURIERS)(packages[c,p]) = 1); % implied constraint with packages. every packages must be shipped by only once

% counting the actual load
constraint forall(c in COURIERS)(actual_load[c] = sum(p in ITEMS where packages[c,p] = 1)(weights[p]));

% LARGE INSTANCES WORK GENERALLY WORSE WITH SIMMETRY BREAKING;
% WITH LITTLE INSTANCES IS FUNDAMENTAL TO PRUNE THE SEARCH.

%        counting the total distance travelled by each courier
constraint forall(c in COURIERS)(total_distance[c] = sum(i in NODES) (distances[i,path[c,i]]));

% Objective: Minimize the maximum distance traveled by any courier

solve minimize max(c in COURIERS)(total_distance[c]);




