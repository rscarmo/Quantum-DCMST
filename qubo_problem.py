from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit.primitives import BackendSampler    
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer import AerSimulator
import time
import networkx as nx
import numpy as np
import math

# import matplotlib
# matplotlib.use('Agg')

class DCMST_QUBO:
    def __init__(self, G, degree_constraints, config, root=0, mixer = None, initial_state = None, seed = 42):
        """
        Initialize the QUBO for the Degree-Constrained Minimum Spanning Tree problem.

        Parameters:
        - G: A networkx.Graph object representing the input graph (weighted).
        - degree_constraints: A dictionary {v: max_degree} for each vertex.
        - root: The root vertex (v0) of the spanning tree.
        """
        self.qubo = QuadraticProgram()
        self.G = G
        self.n = G.number_of_nodes()
        self.degree_constraints = degree_constraints
        self.root = root
        self.config = config
        self.mixer = mixer
        self.initial_state = initial_state
        self.seed = seed
        # Define the penalty coefficient
        num_vertices = self.n
        m = max(data['weight'] for _, _, data in self.G.edges(data=True))
        self.P_I = (num_vertices - 1) * m + 1        
        print('P_I', self.P_I)

    def configure_variables(self):
        """Define variables for edge inclusion (e_{u,v}), order (x_{u,v}), and degree counters (z_{v,i})."""
        # Edge inclusion variables
        for u, v, w in self.G.edges(data='weight'):
            var_name_e = f'e_{u}_{v}'
            self.qubo.binary_var(name=var_name_e)

            # Only one direction for edges involving the root
            if u != self.root and v != self.root:
                var_name_e_reverse = f'e_{v}_{u}'
                self.qubo.binary_var(name=var_name_e_reverse)

        # Ordering variables x_{u,v}
        for u in range(1, self.n):
            if u == self.root:
                continue
            for v in range(u + 1, self.n):
                if v == self.root:
                    continue
                var_name_x = f'x_{u}_{v}'
                self.qubo.binary_var(name=var_name_x)

        # Degree counter variables z_{v,i}
        max_degree = self.degree_constraints.get(v, self.n - 1)        
        binary_bits = int(np.ceil(np.log2(max_degree+1)))
        for v in self.G.nodes():
            for i in range(binary_bits):
                var_name_z = f'z_{v}_{i}'
                self.qubo.binary_var(name=var_name_z)

    def define_objective_function(self):
        """Minimize the total weight of selected edges."""
        objective_terms = {}

        # Root edges: Only consider one direction (v0 -> u)
        for u in self.G.neighbors(self.root):
            edge_var = f'e_{self.root}_{u}'
            weight = self.G[self.root][u]['weight']
            objective_terms[edge_var] = weight

        # Non-root edges: Include both directions (u -> v and v -> u)
        for u, v, w in self.G.edges(data='weight'):
            if u != self.root and v != self.root:
                edge_var_uv = f'e_{u}_{v}'
                edge_var_vu = f'e_{v}_{u}'
                objective_terms[edge_var_uv] = w
                objective_terms[edge_var_vu] = w

        # Add the linear objective terms
        self.qubo.minimize(linear=objective_terms)

    def add_penalty_terms(self):
        # 1. Acyclicity via Topological Ordering Penalty Terms
        for u in range(1, self.n):
            for v in range(u + 1, self.n):
                for w in range(v + 1, self.n):
                    if u == self.root or v == self.root or w == self.root:
                        continue

                    # Define variable names
                    var_name_x_uv = f'x_{u}_{v}'
                    var_name_x_vw = f'x_{v}_{w}'
                    var_name_x_uw = f'x_{u}_{w}'

                    # Ensure all required variables exist
                    variables = [var_name_x_uv, var_name_x_vw, var_name_x_uw]
                    if not all(var in self.qubo.variables_dict for var in variables):
                        continue

                    # Initialize linear and quadratic terms
                    linear_terms = {}
                    quadratic_terms = {}

                    # Add linear term: +P_I * x_{u,w}
                    linear_terms[var_name_x_uw] = linear_terms.get(var_name_x_uw, 0) + self.P_I

                    # Add quadratic terms
                    # +P_I * x_{u,v}x_{v,w}
                    pair_uv_vw = tuple(sorted([var_name_x_uv, var_name_x_vw]))
                    quadratic_terms[pair_uv_vw] = quadratic_terms.get(pair_uv_vw, 0) + self.P_I

                    # -P_I * x_{u,v}x_{u,w}
                    pair_uv_uw = tuple(sorted([var_name_x_uv, var_name_x_uw]))
                    quadratic_terms[pair_uv_uw] = quadratic_terms.get(pair_uv_uw, 0) + (- self.P_I)

                    # -P_I * x_{u,w}x_{v,w}
                    pair_uw_vw = tuple(sorted([var_name_x_uw, var_name_x_vw]))
                    quadratic_terms[pair_uw_vw] = quadratic_terms.get(pair_uw_vw, 0) + (- self.P_I)

                    # Add all terms to the QUBO
                    for var, coeff in linear_terms.items():
                        self.qubo.minimize.linear[var] = self.qubo.minimize.linear.get(var, 0) + coeff

                    for pair, coeff in quadratic_terms.items():
                        self.qubo.minimize.quadratic[pair] = self.qubo.minimize.quadratic.get(pair, 0) + coeff


        # 5. Alignment Constraint (Term (ii))
        for (u, v) in self.G.edges():
            if u < v and u != self.root and v != self.root:
                var_name_e_uv = f'e_{u}_{v}'
                var_name_e_vu = f'e_{v}_{u}'
                var_name_x_uv = f'x_{u}_{v}'

                # Ensure all required variables exist
                variables = [var_name_e_uv, var_name_e_vu, var_name_x_uv]
                if not all(var in self.qubo.variables_dict for var in variables):
                    continue

                # Add linear term: +P_I * e_{u,v}
                self.qubo.minimize.linear[var_name_e_uv] = self.qubo.minimize.linear.get(var_name_e_uv, 0) + self.P_I

                # Add quadratic term: -P_I * e_{u,v}x_{u,v}
                pair = tuple(sorted([var_name_e_uv, var_name_x_uv]))
                self.qubo.minimize.quadratic[pair] = self.qubo.minimize.quadratic.get(pair, 0) + (-self.P_I)

                # Add quadratic term: +P_I * e_{v,u}x_{u,v}
                pair = tuple(sorted([var_name_e_vu, var_name_x_uv]))
                self.qubo.minimize.quadratic[pair] = self.qubo.minimize.quadratic.get(pair, 0) + self.P_I

    def add_constraints(self):
        """Add constraints to enforce the DCMST problem conditions."""
        # 1. Edge count constraint: sum of selected edges = n - 1
        # edge_count_constraint = {
        #     f'e_{u}_{v}': 1 for u, v in self.G.edges()
        # }
        # self.qubo.linear_constraint(
        #     linear=edge_count_constraint,
        #     sense='==',
        #     rhs=self.n - 1,
        #     name='edge_count_constraint'
        # )

        # 2. Acyclicity constraints using ordering variables x_{u,v}
        # These are now handled in define_penalty_terms()        


        # 3. Edge alignment constraints: ensure consistency between e_{u,v} and x_{u,v}
        # These are now handled in define_penalty_terms() 

        """Add connectivity constraints (Constraint iii) as linear constraints."""
        for v in self.G.nodes():
            if v == self.root:
                continue  # Skip the root node

            # Sum of incoming edges to v
            incoming_edges = {
                f'e_{u}_{v}': 1 for u in self.G.neighbors(v)
            }

            # Add linear constraint: sum of incoming edges = 1
            self.qubo.linear_constraint(
                linear=incoming_edges,
                sense='==',
                rhs=1,
                name=f'connectivity_constraint_{v}'
            )

        """Add degree constraints (Constraint iv) as linear constraints."""
        for v in self.G.nodes():
            # Get the maximum allowed degree for v (or default)
            max_degree = self.degree_constraints.get(v, self.n - 1)

            # Number of standard binary bits
            binary_bits = int(math.ceil(np.log2(max_degree + 1)))

            # -----------------------------
            #  1) Construct z_v as:
            #     z_v = sum_{i=0}^{k-1} 2^i z_{v,i} + (Delta+1 - 2^k)*z_{v,k}
            # -----------------------------
            all_terms = {}

            # (A) Standard bits (0..k-1)
            for i in range(binary_bits-1):
                bit_name = f"z_{v}_{i}"
                # Coefficient +2^i
                all_terms[bit_name] = all_terms.get(bit_name, 0.0) + (2**i)

            # (B) Special last bit (kth bit):
            #     (Delta+1 - 2^k) * z_{v,k}
            special_bit_name = f"z_{v}_{binary_bits-1}"
            all_terms[special_bit_name] = all_terms.get(special_bit_name, 0.0) + (
                (max_degree + 1) - 2**(binary_bits-1)
            )

            # -----------------------------
            #  2) Subtract edges so that
            #     [z_v] - [sum of edges] = 0
            # -----------------------------

            if v == self.root:
                # Root node: only outgoing edges e_{root,u}
                for u in self.G.neighbors(self.root):
                    edge_name = f"e_{self.root}_{u}"
                    if edge_name in self.qubo.variables_index:
                        # Subtract 1.0 for each edge
                        all_terms[edge_name] = all_terms.get(edge_name, 0.0) - 1.0

            else:
                # Non-root node: consider both incoming and outgoing edges
                for u in self.G.neighbors(v):
                    edge_in_name = f"e_{u}_{v}"  # incoming edge
                    if edge_in_name in self.qubo.variables_index:
                        all_terms[edge_in_name] = all_terms.get(edge_in_name, 0.0) - 1.0

                    edge_out_name = f"e_{v}_{u}"  # outgoing edge
                    if edge_out_name in self.qubo.variables_index:
                        all_terms[edge_out_name] = all_terms.get(edge_out_name, 0.0) - 1.0

            # -----------------------------
            #  3) Enforce equality: z_v - sum_of_edges = 0
            # -----------------------------
            self.qubo.linear_constraint(
                linear=all_terms,
                sense='==',
                rhs=0.0,
                name=f'degree_constraint_{v}'
            )
         

    def configure_backend(self):
        if self.config.SIMULATION == "True":
            print("Proceeding with simulation...")
            backend = AerSimulator()
            backend.set_options(seed_simulator=self.seed)
        else:
            print("Proceeding with IBM Quantum hardware...")
            service = QiskitRuntimeService(channel='ibm_quantum', token=self.config.QXToken)
            backend = service.least_busy(n_qubits=127, operational=True, simulator=False)
            print(f"Connected to {backend.name}!")
        return backend             

    def solve_problem(self, optimizer, p=1):
        # Convert the problem with constraints into an unconstrained QUBO
        converter = QuadraticProgramToQubo(penalty=self.P_I)
        qubo = converter.convert(self.qubo)
        
        # Print the Ising model
        print(qubo.to_ising())

        qubo_ops, offset = qubo.to_ising()

        backend = self.configure_backend()
        sampler = BackendSampler(backend=backend)

        # sampler = Sampler(
        #     session= backend.service, 
        #     options={"backend": backend.name}  # or whichever device name
        # )        

        # Define a callback function to track progress
        def callback(eval_count, params, mean, std):
            print(f"Eval count: {eval_count}, Parameters: {params}, Mean + offset: {mean + offset}, Std: {std}")

        # Set up QAOA with the callback

        np.random.seed(self.seed)

        # Define the seed that will be used in the optimization process 
        algorithm_globals.random_seed = self.seed

        # Generate initial parameters using the seed
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * p)        

        if (self.mixer is not None) and  (self.initial_state is not None):
            qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=p, initial_point=initial_params, 
                            mixer=self.mixer, initial_state=self.initial_state ,callback=callback)
        else:
            qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=p, initial_point=initial_params, callback=callback)
        qaoa = MinimumEigenOptimizer(qaoa_mes, penalty=self.P_I)

        start_time = time.time()
        qaoa_result = qaoa.solve(self.qubo)
        end_time = time.time()

        self.execution_time = end_time - start_time
        self.solution = qaoa_result.variables_dict
        print(f"Execution time: {self.execution_time} seconds")
        print(qaoa_result.prettyprint())

        return qaoa_result.samples

    def print_number_of_qubits(self):
        """
        Calculate and print the number of qubits used in the problem.
        This is determined by the total number of binary variables in the QUBO.
        """
        num_qubits = len(self.qubo.variables)
        print(f"Number of qubits required: {num_qubits}")

