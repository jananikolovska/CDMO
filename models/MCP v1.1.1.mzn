%to try:
% is it necessary to keep all those double implications in channeling constraints?

% try binpacking and the other to see the best

% implement random or domwdeg on assign

include "globals.mzn";

%       PARAMETERS
int: m; % Number of couriers
int: n; % Number of packages
array [1..m] of int: max_load; % Maximum load capacity for each courier
array [1..n] of int: weights;  % Weights for each package
array[1..n + 1, 1..n + 1] of int: distances;  % Distance matrix (n+1)x(n+1)

%       SETS
set of int: ITEMS = 1..n;
set of int: COURIERS = 1..m;
set of int: NODES = 1..n+1;

%       DECISION VARIABLES
array [ITEMS] of var COURIERS: assign; % Assignment array: each package assigned to a courier
array[COURIERS, NODES] of var 1..(n+1): path;  % Path matrix for each courier

%        AUXILIARY VARIABLES
array[1..m,1..n] of var 0..1: packages; % packages[c,i] = 1 means packages i is delivered by c.
array[1..m] of var int: actual_load;
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
constraint alldifferent(path[COURIERS,n+1]); % all the couriers must deliver at least a package!
constraint forall(c in COURIERS)(path[c,n+1]!=n+1);

%        the maximum loads must be respected for each courier, TRY WHICH ONE PERFORM BETTER
constraint forall(c in COURIERS)(sum(i in ITEMS where path[c,i]!=i)(weights[i]) <= max_load[c]);
% constraint bin_packing_capa(max_load, assign, weights); % global version

%        CHANNELING constraints between path and assign
constraint forall(c in 1..m)(forall(i in 1..n)(
    (assign[i] != c <-> path[c, i] = i) 
    /\ (assign[i] = c <-> path[c, i] != i) 
    )
    /\    path[c, n+1] != n+1 
);

%%%%%%%%%%%%%% WITHOUT PACKAGES AND SYMMETRY BREAKING IT WAS WORKING ALSO ON HARDER INSTANCES

%        CHANNELING constraints between assign and packages.
constraint forall(c in 1..m)(
    forall(i in 1..n)(
      (assign[i] = c <-> packages[c,i] = 1) 
      /\(assign[i] != c <-> packages[c,i] = 0)
    )
);

%        CHANNELING constraints between path and packages
constraint forall(c in 1..m)(
    forall(i in 1..n)(
      (path[c,i] != i <-> packages[c,i] = 1) 
      /\(path[c,i] = i <-> packages[c,i] = 0)
    )
);

%        OTHER PACKAGES CONSTRAINTS

% if every courier sends only a package all the columns of packages must be different (SOMETIMES USEFUL, SOMETIMES NOT ON LARGE INSTANCES)
% constraint alldifferent([assign[i]|i in 1..n]) -> forall (p in 1..n)( % CONSTRAINT NUMBER1
%     alldifferent([packages[c,p]|c in 1..m])
% );


constraint forall(p in 1..n)(sum(c in 1..m)(packages[c,p]) = 1); % implied constraint with packages. every packages must be shipped by only once

% counting the load
constraint forall(c in 1..m)(actual_load[c] = sum(p in 1..n where packages[c,p] = 1)(weights[p]));

% LARGE INSTANCES(11) WORK WORSE WITH SIMMETRY BREAKING;
% WITH LITTLE INSTANCES IS FUNDAMENTAL TO PRUNE THE SEARCH.

%        SYMMETRY BREAKING CONSTRAINTS

%        two couriers with the same maximum load can be exchanged
% constraint forall(c1,c2 in 1..m where c1<c2) (
%     (max_load[c1] = max_load[c2])
%     ->(
%     lex_lesseq([assign[p] = c1 | p in 1..n], [assign[p] = c2 | p in 1..n]) 
%     )
% );

%        two couriers that haven't filled their capacity can be exchanged (too expensive to compute on large instances. no solution found)
% constraint forall(c1,c2 in 1..m where c1<c2) (
%     ((actual_load[c1] <= max_load[c2])/\ (actual_load[c2] <= max_load[c1])) 
% %     (max([actual_load[c1],actual_load[c2]])<min([max_load[c1],max_load[c2]])) % TO STUDY WHICH ONE IS BETTER
%     ->(
%     lex_lesseq([assign[p] = c1 | p in 1..n], [assign[p] = c2 | p in 1..n]) 
%     )
% );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%        counting the total distance travelled by each courier
constraint forall(c in COURIERS)(total_distance[c] = sum(i in NODES) (distances[i,path[c,i]]));

% solve minimize max(c in COURIERS)(total_distance[c])


% Objective: Minimize the maximum distance traveled by any courier
solve 
  :: seq_search([
  int_search(assign,dom_w_deg, indomain_random),
  int_search(path, dom_w_deg, indomain_random), %dom_w_deg may be the best     
])
:: relax_and_reconstruct(assign,95) 
::restart_luby(100)
minimize max(c in COURIERS)(total_distance[c]);

% till now dom_w_deg with indomain_random and r&r(95) work best


