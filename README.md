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
usage: cp.py [-h] [--models {all,sym,lns,plain,custom}] [--instances INSTANCES] [--solvers {all,gecode,chuffed}] [--save SAVE] [--results RESULTS] [--time-limit TIME_LIMIT]
                        
Solve problems with specified solver and instances using CP.

optional arguments:
  -h, --help            show this help message and exit
  --models {all,sym,lns,plain,custom}, -m {all,sym,lns,plain,custom}
                        Specify the model to use. Options: all, sym, lns, plain, custom. Default: all.
                        If custom is selected, the system will ask to type the name of a specific model (e.g: sym.mzn)
  --instances INSTANCES, -i INSTANCES
                        An integer or a list of integers specifying selected instances. Default: "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21".
  --solvers {all,gecode,chuffed}, -s {all,gecode,chuffed}
                        Specify the solver to use. Options: all, gecode, chuffed. Default: all
  --save SAVE, -sv SAVE
                        Enable saving in JSON format. (select: True or False)
  --results RESULTS, -r RESULTS
                        Specify where do you want to save the results. Default: results
  --time-limit TIME_LIMIT, -tl TIME_LIMIT
                        Time limit for the program in seconds. Must be an integer. Default: 300.
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

Run command: `python3 mip.py` <br/><br/>
Help menu displaying changeable parameters:

```
usage: mip.py [-h] [--instances INSTANCES] [--results RESULTS] [--folder-name FOLDER_NAME]
              [--selected SELECTED] [--time-limit TIME_LIMIT] [--print-summary]

A MIP solver script with customizable arguments.

optional arguments:
  -h, --help            show this help message and exit
  --instances INSTANCES, -i INSTANCES
                        Path to the folder containing instance files. Default = "./instances/"
  --results RESULTS, -r RESULTS
                        Path to the base folder where results will be stored. Default = "./results/"
  --folder-name FOLDER_NAME, -f FOLDER_NAME
                        Name of the subfolder to store results. Default: "MIP".
  --selected SELECTED, -s SELECTED
                        An integer or a list of integers specifying selected instances.
                        Default: "1,2,3,4,5,6,7,8,9,10".
  --time-limit TIME_LIMIT, -t TIME_LIMIT
                        Time limit for solving each instance in seconds. Must be an integer.
                        Default: 300.
  --print-summary, -p   Enable verbose output. Default: False.
```



### **Run check_solutions.py** on setup-ed docker <br/>
---
`python3 check_solution.py instances results/`


