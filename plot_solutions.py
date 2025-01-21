import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
import itertools
import numpy as np
import math

def format_qaoa_samples(samples):
    bitstring_counts = {}
    for sample in samples:
        bitstring = ''.join(str(int(v)) for v in sample.x)
        # bitstring = bitstring[::-1]

        if bitstring in bitstring_counts:
            bitstring_counts[bitstring] += 1
        else:
            bitstring_counts[bitstring] = 1


    return bitstring_counts


def interpret_solution(bitstring, adj_matrix, N, Delta):
    """
    Interprets the solution (bitstring) and reconstructs the MST graph,
    including reading out the z variables that encode node degrees.

    Args:
        bitstring: Solution represented as a binary array (or list) of length n_qubits.
        adj_matrix: Adjacency matrix of the graph.
        N: Number of nodes.
        Delta: Maximum degree constraint (used to decode z_v).
    
    Returns:
        mst_edges: List of edges in the MST with weights, or None if invalid.
        z_values: List of decoded z_v for each node v.
    """

    # print(bitstring)

    # Initialize MST structures
    mst_edges = []
    mst_graph = nx.Graph()
    mst_graph.add_nodes_from(range(N))  # Add all nodes (0 to N-1)
    added_edges = set()
    root = 0  # Assume node 0 is the root

    qubit_index = {}
    idx = 0
    # Step 1: Build qubit_index mapping for all variables
    # e_{u,v} and e_{v,u}
    for u in range(N):
        for v in range(u+1, N):
            if adj_matrix[u, v] > 0:
                if u == root:
                    qubit_index[('e', u, v)] = idx
                    idx += 1
                else:
                    qubit_index[('e', u, v)] = idx
                    idx += 1    
                    qubit_index[('e', v, u)] = idx
                    idx += 1                                     

    # x_{u,v} (ordering variables)
    for u in range(1, N):
        for v in range(u + 1, N):
            qubit_index[('x', u, v)] = idx
            idx += 1
    
    # z_v (binary-encoded degree variables)
    binary_degree_size = int(math.ceil(np.log2(Delta + 1)))
    for v in range(N):
        for i in range(binary_degree_size):
            qubit_index[('z', v, i)] = idx
            idx += 1

    # 2) Decode z_v properly
    z_values = [0]*N
    k = binary_degree_size - 1  # if the total #bits = k+1
    for v in range(N):
        z_v = 0
        # sum_{i=0 to k-1} 2^i * z_{v,i} 
        # + (Delta+1 - 2^k) * z_{v,k}

        for i in range(k):  # 0..k-1
            bit_val = int(bitstring[ qubit_index[('z', v, i)] ])
            z_v += (2**i)*bit_val

        # last bit (i = k)
        last_bit_val = int(bitstring[ qubit_index[('z', v, k)] ])
        z_v += (Delta + 1 - 2**k)*last_bit_val

        z_values[v] = z_v

    # print(qubit_index)

    # 2. Reconstruct edges based on e_{u,v} and x_{u,v}

    for u, v in itertools.combinations(range(N), 2):
        if adj_matrix[u, v] > 0 and v!=0 :
            # Case 1: Root edges (only consider e_{0, v})
            if u == root:
                e_uv = bitstring[qubit_index[('e', u, v)]]  # Edge 0->v
                if e_uv == '1' and (u, v) not in added_edges:
                    weight = adj_matrix[u, v]
                    mst_edges.append((u, v, weight))
                    mst_graph.add_edge(u, v, weight=weight)
                    added_edges.add((u, v))
                continue

            # Case 2: Non-root edges
            e_uv = bitstring[qubit_index[('e', u, v)]]  # Edge u->v
            e_vu = bitstring[qubit_index[('e', v, u)]]  # Edge v->u
            try:
                x_uv = bitstring[qubit_index[('x', u, v)]]  # Order: u->v
            except:
                x_uv = '0'

            # If e_uv == 1 and x_uv == 1, that means the chosen edge is u->v
            # and if e_vu == 1 and x_uv == 0, that means the chosen edge is v->u
            # (depending on your exact MST constraint logic).
            
            # Simplified check: if exactly one direction is chosen
            # and the ordering is consistent with that direction.
            if ((e_uv == '1' and x_uv == '1') or
                (e_vu == '1' and x_uv == '0')) and (e_uv != e_vu) and ((u, v) not in added_edges):
                weight = adj_matrix[u, v]
                mst_edges.append((u, v, weight))
                mst_graph.add_edge(u, v, weight=weight)
                added_edges.add((u, v))

    # 3. Check if we formed a valid MST
    #    (connected and exactly N-1 edges)  
    if nx.is_connected(mst_graph) and mst_graph.number_of_edges() == (N - 1):
        # Optional: Verify the actual degree vs. z_v
        actual_degrees = [mst_graph.degree[v] for v in range(N)]
        for v in range(N):
            if actual_degrees[v] > z_values[v]:
                # print(f"Warning: Node {v} has actual degree {actual_degrees[v]} "
                #       f"but z_v = {z_values[v]}. (Mismatch!)")
                return None
            if actual_degrees[v] > Delta:
                # print(f"Warning: Node {v} exceeds Delta. "
                #       f"Deg={actual_degrees[v]}, Delta={Delta}")
                return None

        return mst_edges #, z_values
    else:
        return None #, z_values

def sample_and_plot_histogram(samples, adj_matrix, N, Delta, interpret_solution_fn, top_n=30):
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
    for bitstring in samples:
        bit_array = "".join([str(int(_)) for _ in bitstring.x])
        # bit_array = bit_array[::-1] 
        mst_solution = interpret_solution_fn(bit_array, adj_matrix, N, Delta)
        if mst_solution is not None:
            valid_solutions.append(tuple(sorted(mst_solution)))  # Normalize for comparison

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
    plt.show()

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
