import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
import itertools
import numpy as np
import math


def sample_to_dict(sample, var_names):
    """
    Convert a Qiskit 'SolutionSample' to a {var_name: bit_value} dict.
    """
    # sample.x is an array/list of 0/1 in the same order as 'var_names'
    # We cast each bit to int just for safety.
    try:
        return {
            name: int(val)
            for name, val in zip(var_names, sample.x)
        }
    except:
        return {
            name: int(val)
            for name, val in zip(var_names, sample)
        }        

def interpret_solution(solution_dict, adj_matrix, N, Delta):
    """
    Interprets the solution (solution_dict) and reconstructs the MST graph,
    including reading out the z_{v,i} variables that encode node degrees.

    Args:
        solution_dict: Dictionary {var_name: 0 or 1} for all variables in the QUBO.
                       For example, {'e_0_1': 1, 'x_1_2': 0, 'z_3_0': 1, ...}.
        adj_matrix: Adjacency matrix of the graph.
        N: Number of nodes.
        Delta: Maximum degree constraint (used to decode z_v).

    Returns:
        mst_edges: List of edges in the MST with weights, or None if invalid.
    """

    # print(solution_dict)

    root = 0  # Assume node 0 is the root
    mst_graph = nx.Graph()
    mst_graph.add_nodes_from(range(N))  # Add all nodes (0 to N-1)
    added_edges = set()

    # 1) Decode each node's degree z_v from the variables "z_{v}_{i}"
    binary_degree_size = int(math.ceil(np.log2(Delta + 1)))
    z_values = [0] * N

    for v in range(N):
        z_v = 0
        # sum_{i=0..k-1} (2^i * z_{v,i}) + (Delta+1 - 2^k)*z_{v,k}
        # if k+1 = binary_degree_size
        k = binary_degree_size - 1
        for i in range(k):
            bit_val = solution_dict.get(f"z_{v}_{i}", 0)
            z_v += (2**i) * bit_val
        # Handle the last bit
        last_bit_val = solution_dict.get(f"z_{v}_{k}", 0)
        z_v += (Delta + 1 - 2**k) * last_bit_val

        z_values[v] = z_v

    # 2) Reconstruct edges based on e_{u,v} and x_{u,v} variables
    mst_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj_matrix[u, v] <= 0:
                continue  # No edge in the original graph

            # Case A: If u == root, we only consider e_{0,v}

            if u == root:
                var_name_uv = f"e_{u}_{v}"  # e_0_v
                e_uv = solution_dict.get(var_name_uv, 0)
                if e_uv == 1 and (u, v) not in added_edges:
                    weight = adj_matrix[u, v]
                    mst_edges.append((u, v, weight))
                    mst_graph.add_edge(u, v, weight=weight)
                    added_edges.add((u, v))
                continue

            # Case B: Non-root edges => e_{u,v}, e_{v,u}, x_{u,v}
            var_name_uv = f"e_{u}_{v}"
            var_name_vu = f"e_{v}_{u}"
            e_uv = solution_dict.get(var_name_uv, 0)
            e_vu = solution_dict.get(var_name_vu, 0)

            # x_{u,v} is the ordering variable
            x_uv_name = f"x_{u}_{v}"
            x_uv = solution_dict.get(x_uv_name, 0)

            # If e_uv == 1 and x_uv == 1 => chosen edge is (u->v)
            # If e_vu == 1 and x_uv == 0 => chosen edge is (v->u)
            # We'll accept it if only one direction is chosen, plus ordering consistency
            if (
                ((e_uv == 1 and x_uv == 1) or (e_vu == 1 and x_uv == 0))
                and (e_uv != e_vu)
                and ((u, v) not in added_edges)
            ):
                weight = adj_matrix[u, v]
                mst_edges.append((u, v, weight))
                mst_graph.add_edge(u, v, weight=weight)
                added_edges.add((u, v))

    # 3) Check if we formed a valid MST
    #    (connected and exactly N-1 edges)
    if nx.is_connected(mst_graph) and mst_graph.number_of_edges() == (N - 1):
        # Optional: Verify the actual degree vs. z_v
        actual_degrees = [mst_graph.degree[v] for v in range(N)]
        for v in range(N):
            if actual_degrees[v] > z_values[v]:
                # Mismatch: MST says we have actual_degrees[v], but z_v is smaller
                # print(f'Actual degree: {actual_degrees[v]}, z_v:{z_values[v]}')
                return None
            if actual_degrees[v] > Delta:
                # The MST itself violates the maximum degree constraint
                return None
        return mst_edges  # Valid MST solution
    else:
        return None  # Not a valid MST


def sample_and_plot_histogram(samples, adj_matrix, N, Delta, interpret_solution_fn, top_n=30, var_names=None):
    """
    Interpret QUBO samples, validate solutions, and plot a histogram of the most sampled valid bitstrings.

    Parameters:
    - samples: Dictionary of bitstrings (keys) and their frequencies (values) from Qiskit.
    - adj_matrix: Adjacency matrix of the graph.
    - N: Number of nodes in the graph.
    - Delta: Maximum degree constraint.
    - interpret_solution_fn: Function to interpret and validate a bitstring as an MST.
    - top_n: Number of most common solutions to display in the histogram.

    Returns:
    - most_common_valid_solutions: List of the most common valid solutions (up to `top_n`).
    """
    # Step 1: Interpret and validate solutions
    valid_solutions = []
    for sample in samples:
        # Convert sample -> dictionary
        solution_dict = sample_to_dict(sample, var_names)
        converted_dict = {
            var.name: val for var, val in solution_dict.items()
        }        

        # Now call the dictionary-based function
        mst_solution = interpret_solution_fn(converted_dict, adj_matrix, N, Delta)
        if mst_solution is not None:
            valid_solutions.append(tuple(sorted(mst_solution)))

    if not valid_solutions:
        print("No valid MST solutions were found.")
        return []

    # Step 2: Count frequencies of valid solutions
    solution_counts = Counter(valid_solutions)

    # Step 3: Extract the top N most common valid solutions
    most_common_valid_solutions = solution_counts.most_common(top_n)

    # Step 4: Plot the histogram
    labels = [f"Solution {i+1}" for i in range(len(most_common_valid_solutions))]
    frequencies = [count for _, count in most_common_valid_solutions]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, frequencies, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} Most Sampled Valid Solutions")
    plt.xlabel("Solutions")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # plt.show()

    # Step 5: Print the details of the most common solutions
    print("\nTop Valid MST Solutions:")
    for i, (solution, frequency) in enumerate(most_common_valid_solutions, start=1):
        cost = sum(adj_matrix[u][v] for u, v, _ in solution)
        print(f"Solution {i}: {solution}\nFrequency: {frequency}\nTotal Cost: {cost:.2f}\n")

    return most_common_valid_solutions


def draw_solution(graph, solution, title="Minimum Spanning Tree Solution"):
    """
    Visualize the solution based on binary variable values (Qiskit-style).
    
    Parameters:
    - solution: Dictionary of binary variable assignments from Qiskit.
    - title: Title of the plot.
    """
    mst_edges = []

    # Extract edges included in the solution
    for (u, v) in graph.G.edges:
        var_name = f'e_{u}_{v}'
        if solution.get(var_name, 0) == 1:
            mst_edges.append((u, v))
        var_name = f'e_{v}_{u}'
        if solution.get(var_name, 0) == 1:
            mst_edges.append((v, u))            

    if mst_edges:
        pos = graph.pos or nx.spring_layout(graph.G)
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(graph.G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(graph.G, pos, edgelist=mst_edges, edge_color='r', width=2)
        nx.draw_networkx_edges(
            graph.G, pos, edgelist=[e for e in graph.G.edges if e not in mst_edges], style='dotted'
        )
        nx.draw_networkx_labels(graph.G, pos, font_size=12, font_color='black')
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print("No edges found in the MST solution. Please check the variable consistency.")
