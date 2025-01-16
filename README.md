# CDMO

**Setup Docker** following the next steps: <br/> 
1. Download and open Docker
2. `docker build -t mcp_cdmo .`
3. `docker run -v "$(pwd):/app" --name cdmo -it mcp_cdmo`

**Run smt.py** on setup-ed docker <br/>
`python3 smt.py --instances instances --results results --folder-name SMT_exmple --selected "1,2,3,4,5,6,7,8,9,10"`

**Run cp.py** on setup-ed docker <br/>
`python3 cp.py` 

**Run mip.py** on setup-ed docker <br/>

**Run check_solutions.py** on setup-ed docker <br/>
`python3 check_solution.py instances results/`


