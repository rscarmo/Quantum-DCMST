from graph import Graph
from qubo_problem import DCMST_QUBO
from plot_solutions import draw_solution, sample_and_plot_histogram, interpret_solution
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from config import Config
import numpy as np


def main():
    # Initial configurations
    config = Config()

    # Example usage
    N = 4
    weight_range = (10, 100)
    seed = 51 #78
    max_degree = 2
    VQE=True

    # Instantiate and create the graph
    graph = Graph(N, weight_range, seed)
    graph.draw()

    # Generate the adjacency matrix
    adj_matrix = np.zeros((N, N), dtype=int)
    for (u, v, data) in graph.G.edges(data=True):
        adj_matrix[u][v] = data['weight']
        adj_matrix[v][u] = data['weight']  # Ensure symmetry    

    # Find the mst using kruskal
    mst_edges, total_weight = graph.find_mst_kruskal()
    graph.plot_mst(mst_edges, title="Kruskal MST")

    # find the dcmst by brute force
    dcmst_edges, dcmst_cost = graph.find_dcmst_brute_force(max_degree)
    if dcmst_edges is not None:
        graph.plot_mst(dcmst_edges, mst_color='red', title=f"DC-MST (max_degree={max_degree})")

    # Create degree_constraints dictionary with the same max degree for all nodes
    degree_constraints = {node: max_degree for node in graph.G.nodes()}

    # Configure and solve the QUBO problem

    # With Warm-Starting
    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, 
    #                           mixer='Warm', initial_state='RY', regularization=0.16,
    #                           fake_backend=False)
    
    # With LocicalX Mixer - This is not working yet
    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, mixer='LogicalX', initial_state='OHE')

    # With mixer X - Standard formulation
    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, fake_backend=False)

    # With mixer X - Standard formulation + redundant conditions
    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, redundancy=True)

    # With VQE 
    qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, VQE=VQE)

    qubo_problem.configure_variables()
    qubo_problem.define_objective_function()
    qubo_problem.add_constraints()

    # Print the number of qubits necessary to solve the problem
    qubo_problem.print_number_of_qubits()

    # Get a reference solution by brute force (only feasible for small problems between 20-25 variables!)
    # bf_solution, bf_cost = qubo_problem.brute_force_solution()    

    optimizer = COBYLA()
    p =3  # QAOA circuit depth

    if not VQE:
        optimal_params = qubo_problem.solve_problem(optimizer, p)
        samples = qubo_problem.qubo_sample(optimal_params)
    else:
        samples = qubo_problem.solve_problem(optimizer, p)


    # It seems to me that the best result considered by qiskit only takes into 
    # account the objective function and constraints, but not  penalties added directly to the QuadraticProgramm.
    # solution = qubo_problem.solution

    # Visualize the solution

    valid_solutions = sample_and_plot_histogram(
        samples,
        adj_matrix=adj_matrix,
        N=N,
        Delta=max_degree,
        interpret_solution_fn=interpret_solution,
        top_n=30,
        var_names=qubo_problem.var_names,
        VQE=VQE
    )    

    # Draw the most sampled
    # draw_solution(graph, solution)

    # print('BEST BITSTRING:', bf_solution)
    # print('BEST COST:', bf_cost)


if __name__ == "__main__":
    main()