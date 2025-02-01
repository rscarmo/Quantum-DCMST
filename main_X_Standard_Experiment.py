from graph import Graph
from qubo_problem import DCMST_QUBO
from plot_solutions import draw_solution, sample_and_plot_histogram, interpret_solution
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from config import Config
import numpy as np
import csv
import pickle
import os
import pandas as pd


def load_cache(cache_file='brute_force_cache.pkl'):
    """
    Load the brute-force solutions cache from a file.

    Parameters:
        cache_file (str): Path to the cache file.

    Returns:
        dict: The loaded cache dictionary.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            solutions_brute_force = pickle.load(f)
        print(f"Loaded cache from {cache_file}.")
    else:
        solutions_brute_force = {}
        print("Initialized empty brute-force cache.")
    return solutions_brute_force

def get_brute_force_solution(qubo_problem, key, cache):
    """
    Retrieve the brute-force solution from cache or compute it if not present.

    Parameters:
        qubo_problem (DCMST_QUBO): The QUBO problem instance.
        key (tuple): Unique identifier for the graph configuration.
        cache (dict): The cache dictionary.

    Returns:
        tuple: (bf_solution (str), bf_cost (float))
    """
    if key in cache:
        print(f"Cache hit for key: {key}")
        return cache[key]
    else:
        print(f"Cache miss for key: {key}. Computing brute-force solution...")
        bf_solution, bf_cost = qubo_problem.brute_force_solution()
        cache[key] = (bf_solution, bf_cost)
        return bf_solution, bf_cost
    
def save_cache(cache, cache_file='brute_force_cache.pkl'):
    """
    Save the brute-force solutions cache to a file.

    Parameters:
        cache (dict): The cache dictionary.
        cache_file (str): Path to the cache file.
    """
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Saved cache to {cache_file}.")


def main():
    # Initial configurations
    config = Config()

    # Load existing cache or initialize a new one
    cache_file = 'brute_force_cache.pkl'
    solutions_brute_force = load_cache(cache_file)

    for max_degree in range(2,4):
        for N in range(3,5):
            if N==3 and max_degree == 3:
                continue
            for i in range(5):
                seed = 51 + i               
                for p in range(1,6):    
                    # Define the unique key for the current graph configuration
                    key = (seed, N, max_degree)

                    # Instantiate and create the graph
                    graph = Graph(N, weight_range = (10, 100), seed=seed)
                    # graph.draw()

                    # Generate the adjacency matrix
                    adj_matrix = np.zeros((N, N), dtype=int)
                    for (u, v, data) in graph.G.edges(data=True):
                        adj_matrix[u][v] = data['weight']
                        adj_matrix[v][u] = data['weight']  # Ensure symmetry    

                    # Find the mst using kruskal
                    mst_edges, total_weight = graph.find_mst_kruskal()
                    # graph.plot_mst(mst_edges, title="Kruskal MST")

                    # find the dcmst by brute force
                    dcmst_edges, dcmst_cost = graph.find_dcmst_brute_force(max_degree)
                    # if dcmst_edges is not None:
                    #     graph.plot_mst(dcmst_edges, mst_color='red', title=f"DC-MST (max_degree={max_degree})")

                    # Create degree_constraints dictionary with the same max degree for all nodes
                    degree_constraints = {node: max_degree for node in graph.G.nodes()}

                    # Configure and solve the QUBO problem

                    # This warm_start=True is for using the built-in function from Qiskit to solve a Qubo problem using warm-start techinique
                    # Here for this kind of problem this does not work, because if we relax the x variables, the problem becomes non-convex. 
                    # So, we have to mantain those variables binary and relax all the others. 

                    # With mixer X - Standard formulation
                    qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, fake_backend=False)

                    # With mixer X - Standard formulation + redundant conditions
                    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, redundancy=True)

                    # With Warm-Starting
                    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, mixer='Warm', initial_state='RY', regularization=0.25)
                    
                    # With LocicalX Mixer - This is not working yet
                    # qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config, mixer='LogicalX', initial_state='OHE')


                    qubo_problem.configure_variables()
                    qubo_problem.define_objective_function()
                    qubo_problem.add_constraints()

                    # Print the number of qubits necessary to solve the problem
                    qubo_problem.print_number_of_qubits()

                    # Retrieve or compute the brute-force solution - feasible for <=20 variables
                    bf_solution, bf_cost = None, None #get_brute_force_solution(qubo_problem, key, solutions_brute_force)                   

                    optimizer = COBYLA()
                    # optimal_params = qubo_problem.solve_problem(optimizer, p)

                    # -------------------------------------------------------------------------------------------
                    # Apenas amostragem usando os parâmetros já otimizados
                    qubo_problem.qubo_build_circuit(p)

                    df_func_X = pd.read_csv('all_params_with_metadata.csv')
                    # Agrupando e mantendo apenas best_params únicos
                    # unique_best_params = (
                    #     df_func_X.groupby(['seed_grafo', 'Num_Nos', 'num_layer', 'Delta'], as_index=False)
                    #     .agg({'best_params': 'first'})  # Mantém o primeiro 'best_params' único por grupo
                    # )
                    # Filtrar o dataframe com base nos valores específicos
                    filtered_row = df_func_X[
                        (df_func_X['seed'] == seed) &
                        (df_func_X['N'] == N) &
                        (df_func_X['p'] == p) &
                        (df_func_X['max'] == max_degree)
                    ]

                    if filtered_row.empty:
                        print("Nenhuma linha encontrada para os critérios fornecidos.")
                        continue

                    # Selecionar apenas as colunas que contenham "params"
                    params_columns = [col for col in df_func_X.columns if "params" in col]
                    
                    # Extrair os valores e converter para um array do NumPy
                    optimal_params = filtered_row[params_columns].values.flatten()

                    # Remover valores NaN
                    optimal_params = optimal_params[~np.isnan(optimal_params)]                    

                    # optimal_params = filtered_row.iloc[0]['best_params']  # Acessa o primeiro (e único) resultado
                    print(f"Parâmetros encontrados: {optimal_params}")
                    # -------------------------------------------------------------------------------------------

                    # optimal_params = np.fromstring(optimal_params.strip("[]"), sep=" ")

                    samples = qubo_problem.qubo_sample(optimal_params)
                    with open("qaoa_dcmst_results_3_4_X_PSO_complete.csv", mode="a", newline="") as file:
                        writer = csv.writer(file)
                        
                        # Escrever o cabeçalho se o arquivo está vazio
                        if file.tell() == 0:
                            writer.writerow(["mixer", "seed_grafo", "num_layer", 
                                            "Num_Nos", "Delta", "FakeKolkataV2", 
                                            "bitstring", "frequence", "correct_solution", "correct_cost"])
                            
                        for key, value in samples.items():
                            writer.writerow(['X', seed, p, N, max_degree, False, key, value, bf_solution, bf_cost]) 


                    # Visualize the solution

                    valid_solutions = sample_and_plot_histogram(
                        samples,
                        adj_matrix=adj_matrix,
                        N=N,
                        Delta=max_degree,
                        interpret_solution_fn=interpret_solution,
                        top_n=30,
                        var_names=qubo_problem.var_names
                    )    

                    if valid_solutions == []:
                        with open("qaoa_dcmst_results_3_4_X_PSO.csv", mode="a", newline="") as file:
                            writer = csv.writer(file)
                            
                            # Escrever o cabeçalho se o arquivo está vazio
                            if file.tell() == 0:
                                writer.writerow(["mixer", "redundancy", "seed_grafo", "num_layer", "MST", "frequence", 
                                                "MST_Rank", "MST_cost", "MST_correct", "correct_cost", 
                                                "best_params", "Num_Nos", "Delta", "FakeKolkataV2"])
                            
                            writer.writerow(['warm', 'N', seed, p, 'Nenhuma solução válida', 0, 0, '', dcmst_edges,
                                            dcmst_cost, '', N, max_degree, False])    
                    else:
                        # Configurar o arquivo CSV e escrever os resultados
                        with open("qaoa_dcmst_results_3_4_X_PSO.csv", mode="a", newline="") as file:
                            writer = csv.writer(file)
                            
                            # Escrever o cabeçalho se o arquivo está vazio
                            if file.tell() == 0:
                                writer.writerow(["mixer", "redundancy", "seed_grafo", "num_layer", "MST", "frequence", 
                                                "MST_Rank", "MST_cost", "MST_correct", "correct_cost", 
                                                "best_params", "Num_Nos", "Delta", "FakeKolkataV2"])
                                
                            # Escrever cada MST mais frequente no CSV
                            for i, (mst, frequency, *_) in enumerate(valid_solutions, start=1):
                                mst_no_tensor = [(u, v, weight.item() if hasattr(weight, "item") else weight) for u, v, weight in mst]
                                cost = sum(adj_matrix[u][v] for u, v, weight in mst_no_tensor)
                                
                                writer.writerow(['warm', 'N', seed, p, mst_no_tensor, frequency, 
                                                i, cost, dcmst_edges, dcmst_cost, optimal_params, N, max_degree, False])

if __name__ == "__main__":
    main()