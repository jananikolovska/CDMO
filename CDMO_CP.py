# hello_world.py
from minizinc import Instance, Model, Solver

# Load the MiniZinc model
model = Model("hello_world.mzn")

# Use the Gecode solver (installed with MiniZinc by default)
gecode = Solver.lookup("gecode")

# Create an instance of the model with the solver
instance = Instance(gecode, model)

# Solve the instance
result = instance.solve()

# Output the result
print("Solution:")
print(f"x = {result['x']}")
print(f"y = {result['y']}")
