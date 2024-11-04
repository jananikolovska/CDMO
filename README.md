# CDMO

### CP
to do 
- ~~run MiniZinc through Docker flow - Jana&Cico~~
- ~~change Docker flow to not need to build every time - Cico~~
- update dummy CP script to an initial implementation of our problem

### SAT

to do
- ~~run script for all instances - Jana~~
**Done** for up until 14, from 10 onwards they all failed and took too long so for (14-21) I put failed ones
- ~~run check_solution.py to test locally -Jana~~ 
**Done** problem with logic of function in CDMO_SAT.py
- run full flow from docker -Jana
- improve logic in CDMO_SAT.py
<br>

**CDMO_SAT.ipynb** - visualization code that can be useful for the report 
 
**Setup Docker** following the next steps: <br/> 
1. Download and open Docker
2. `docker build -t z3_solver .`
3. `docker run  -v "$(pwd):/app" -it z3_solver`

**Run CDMO_SAT.py** on setup-ed docker <br/>
`python3 CDMO_SAT instances results/SAT`

**Run check_solutions.py** on setup-ed docker <br/>
`python3 check_solution.py instances results/SAT`

**Run cp.py** on setup-ed docker <br/>
`python3 cp.py <instance.dat>`

