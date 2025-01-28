# Multiple courier satisfaction problem
In this report, we will explore how the problem of the Multiple Couriers Planning (MCP) is modeled
and discuss various approaches to address it: Constraint Programming (CP), Satisfiability Modulo
Theories (SMT) and Mixed-Integer Linear Programming (MIP). The project was completed by a group
of three, with the modeling phase and boundary formula derivation done collaboratively. 

_Find report in repository: REPORT.pdf_

### **Setup Docker** following the next steps: <br/> 
---
1. Download and open Docker
2. `docker build -t mcp_cdmo .`
3. `docker run -v "$(pwd):/app" --name cdmo -it mcp_cdmo`

### **Run cp.py** on setup-ed docker <br/>
---
Run command: `python3 cp.py`  <br/><br/>
Help menu displaying changeable parameters:
```
usage: cp.py [-h] [--mode {normal,superuser}] [--models {all,sym,lns,plain,custom}] [--instances {all,soft,hard}] [--solvers {all,gecode,chuffed}]
                        Specify the solver to use. Options: all, gecode, chuffed.
  --save {true,false}   Specify whether you want the results saved in a JSON.
  --results RESULTS     Specify where do you want to save the results.
```


### **Run smt.py** on setup-ed docker <br/>
---
Run command: `python3 smt.py` <br/><br/>
Help menu displaying changeable parameters:
```
usage: smt.py [-h] [--instances-folder INSTANCES_FOLDER] [--results-folder RESULTS_FOLDER] [--results-subfolder-name RESULTS_SUBFOLDER_NAME]
              [--instances INSTANCES] [--time-limit TIME_LIMIT]

Solve problems with specified solver and instances using SMT.

optional arguments:
  -h, --help            show this help message and exit
  --instances-folder INSTANCES_FOLDER, -if INSTANCES_FOLDER
                        Path to the folder containing instance files. Default = "instances"
  --results-folder RESULTS_FOLDER, -rf RESULTS_FOLDER
                        Path to the base folder where results will be stored. Default = "results"
  --results-subfolder-name RESULTS_SUBFOLDER_NAME, -sf RESULTS_SUBFOLDER_NAME
                        Name of the subfolder to store results. Default: "SMT".
  --instances INSTANCES, -i INSTANCES
                        An integer or a list of integers specifying selected instances. Default:
                        "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21").
  --time-limit TIME_LIMIT, -tl TIME_LIMIT
                        Time limit for the program in seconds. Must be an integer. Default: 300.
```
### **Run mip.py** on setup-ed docker <br/>
---

### **Run check_solutions.py** on setup-ed docker <br/>
---
`python3 check_solution.py instances results/`


