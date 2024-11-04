import numpy as np
import matplotlib.pyplot as plt
# import networkx as nx
import pandas as pd
import time
import random

def generate_hex_colors(size):
    hex_colors = []
    for _ in range(size):
        # Generate a random color and format it as a hex string
        hex_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        hex_colors.append(hex_color)
    return hex_colors

def visualize_solution(num_couriers, num_items, routes):
  # Create a graph
  G = nx.Graph()

  # Assign colors to different couriers
  colors = generate_hex_colors(len(routes))

  # Add edges for each courier route
  for i, route in enumerate(routes):
      # Add edges from depot to first customer, between customers, and back to depot
      G.add_edge('Depot', f'Customer {route[0]}', color=colors[i])
      for j in range(len(route) - 1):
          G.add_edge(f'Customer {route[j]}', f'Customer {route[j+1]}', color=colors[i])
      G.add_edge(f'Customer {route[-1]}', 'Depot', color=colors[i])

  # Get positions for nodes (using a spring layout to prevent overlap)
  pos = nx.spring_layout(G, seed=42)  # Set seed for reproducibility

  # Extract edge colors
  edge_colors = [G[u][v]['color'] for u, v in G.edges()]

  # Draw the graph
  nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10,
          edge_color=edge_colors, width=2)

  # Display the plot
  plt.title('Visualize Solution with NetworkX')
  plt.grid(True)
  plt.show()

# Function to read the .dat file and store first four lines
def read_dat_file(file_path, print_summary = False):
    with open(file_path, 'r') as file:
        # Read the first four lines
        m = int(file.readline().strip())  # First line
        n = int(file.readline().strip())  # Second line
        load_sizes = list(map(int, file.readline().strip().split()))  # Third line
        item_sizes = list(map(int, file.readline().strip().split()))  # Fourth line

        # Read the rest of the data into a list (if needed)
        data = []
        for line in file:
            data.append(list(map(int, line.strip().split())))

    if print_summary:
        print("############SUMMARY############")
        print(f"num_couriers: {m}")
        print(f"num_items: {n}")
        print(f"Load sizes: {load_sizes}")
        print(f"Item sizes: {item_sizes}")
        print("Distance matrix:")
        print(data)
        print("############END############\n")

    return m, n, load_sizes, item_sizes, np.array(data)