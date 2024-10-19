int: m;  % Number of couriers
int: n;  % Number of items

%-----VARIABLES------

% Load capacities for each courier
array[1..m] of int: l;

% Sizes of each item
array[1..n] of int: s;

% Distance matrix (including origin point n+1)
array[1..n+1, 1..n+1] of int: D;

% Decision variables
array[1..m, 1..n] of var 0..1: x;   % x[i,j] = 1 if courier i takes item j

% y[i,j,k] = 1 if courier i travels from point j to point k (where 1..n are items, n+1 is origin)
array[1..m, 1..n+1, 1..n+1] of var 0..1: y;


%d[i] = total distance traveled by courier i
array[1..m] of var int: D_i;

% Maximum distance traveled by any courier
var int: D_max;


%------CONSTRAINTS------

% Item assignment constraint: each item must be assigned to exactly one courier
constraint
  forall(j in 1..n) (
    sum(i in 1..m) (x[i,j]) = 1
  );
  
% Load constraint: each courier's load must not exceed their capacity
constraint
  forall(i in 1..m) (
    sum(j in 1..n) (x[i,j] * s[j]) <= l[i]
  );

% Ensure each courier starts and ends at the origin (point n+1)
constraint
  forall(i in 1..m) (
    sum(k in 1..n) (y[i,n+1,k]) = 1 /\
    sum(j in 1..n) (y[i,j,n+1]) = 1
  );

% Subtour elimination and route consistency
constraint
  forall(i in 1..m, j in 1..n) (
    sum(k in 1..n+1) (y[i,j,k]) = sum(k in 1..n+1) (y[i,k,j])
  );

% Distance calculation for each courier

constraint
  forall(i in 1..m) (
    D_i[i] = sum(j in 1..n+1, k in 1..n+1) (D[j,k] * y[i,j,k])
  );

% Maximum distance constraint
constraint
  forall(i in 1..m) (
    D_i[i] <= D_max
  );

% Objective: Minimize the maximum distance traveled by any courier
solve minimize D_max;


