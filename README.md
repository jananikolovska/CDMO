# CDMO

### SAT

to do
- run script for all instances - Jana
- run check_solution.py to test locally -Jana
- run full flow from docker -Jana
<br>

**CDMO_SAT.ipynb** - visualization code that can be useful for the report 
 
**Setup Docker** following the next steps: <br/> 
1. Download and open Docker
2. `docker build -t z3_solver .`
3. `docker run -it z3_solver`

**Run CDMO_SAT.py** on setup-ed docker <br/>
`python3 CDMO_SAT instances results/SAT`

**Run check_solutions.py** on setup-ed docker <br/>
`python3 check_solution.py instances results/SAT`

