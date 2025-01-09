from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.primitives import BackendSampler
from qiskit_aer import AerSimulator
import time
import networkx as nx

class DCMST_QUBO:
    def __init__(self, G, degree_constraints, root=0):
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
        for v in self.G.nodes():
            max_degree = self.degree_constraints.get(v, self.n - 1)
            num_degree_vars = max_degree.bit_length()
            for i in range(num_degree_vars):
                var_name_z = f'z_{v}_{i}'
                self.qubo.binary_var(name=var_name_z)

    def define_objective_function(self):
        """Minimize the total weight of selected edges."""
        objective_terms = {
            f'e_{u}_{v}': w for u, v, w in self.G.edges(data='weight')
        }
        self.qubo.minimize(linear=objective_terms)

    def add_constraints(self):
        """Add constraints to enforce the DCMST problem conditions."""
        # 1. Edge count constraint: sum of selected edges = n - 1
        edge_count_constraint = {
            f'e_{u}_{v}': 1 for u, v in self.G.edges()
        }
        self.qubo.linear_constraint(
            linear=edge_count_constraint,
            sense='==',
            rhs=self.n - 1,
            name='edge_count_constraint'
        )

        # 2. Acyclicity constraints using ordering variables x_{u,v}
        for u in range(1, self.n):
            if u == self.root:
                continue
            for v in range(u + 1, self.n):
                if v == self.root:
                    continue
                # Add acyclicity constraint: x_{u,v} + x_{v,u} <= 1
                var_name_x_uv = f'x_{u}_{v}'
                var_name_x_vu = f'x_{v}_{u}'
                self.qubo.linear_constraint(
                    linear={var_name_x_uv: 1, var_name_x_vu: 1},
                    sense='<=',
                    rhs=1,
                    name=f'acyclicity_{u}_{v}'
                )

        # 3. Edge alignment constraints: ensure consistency between e_{u,v} and x_{u,v}
        for u, v in self.G.edges():
            if u != self.root and v != self.root:
                var_name_e_uv = f'e_{u}_{v}'
                var_name_e_vu = f'e_{v}_{u}'
                var_name_x_uv = f'x_{u}_{v}'

                # Add alignment constraint: e_{u,v} = x_{u,v}
                self.qubo.linear_constraint(
                    linear={var_name_e_uv: 1, var_name_x_uv: -1},
                    sense='==',
                    rhs=0,
                    name=f'edge_alignment_{u}_{v}'
                )

        # 4. Degree constraints: degree of each vertex <= max_degree
        for v in self.G.nodes():
            max_degree = self.degree_constraints.get(v, self.n - 1)
            degree_edges = {
                f'e_{u}_{v}': 1 for u in self.G.neighbors(v)
            }
            degree_counter_vars = {
                f'z_{v}_{i}': 2**i for i in range(max_degree.bit_length())
            }

            self.qubo.linear_constraint(
                linear={**degree_edges, **degree_counter_vars},
                sense='<=',
                rhs=max_degree,
                name=f'degree_constraint_{v}'
            )

    def configure_backend(self):
        if self.config.SIMULATION == "True":
            print("Proceeding with simulation...")
            backend = AerSimulator()
        else:
            print("Proceeding with IBM Quantum hardware...")
            service = QiskitRuntimeService(channel='ibm_quantum', token=self.config.QXToken)
            backend = service.least_busy(n_qubits=127, operational=True, simulator=False)
            print(f"Connected to {backend.name}!")
        return backend             

    def solve_problem(self, optimizer, p=1):
        # Convert the problem with constraints into an unconstrained QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(self.qubo)
        
        # Now you can print the Ising model and continue
        print(qubo.to_ising())
        
        backend = self.configure_backend()
        sampler = BackendSampler(backend=backend)
        optimizer = COBYLA()
        qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
        qaoa = MinimumEigenOptimizer(qaoa_mes)

        start_time = time.time()
        qaoa_result = qaoa.solve(self.qubo)
        end_time = time.time()

        self.execution_time = end_time - start_time
        self.solution = qaoa_result.variables_dict
        print(f"Execution time: {self.execution_time} seconds")
        print(qaoa_result.prettyprint())

        return qaoa_result.samples
       

# Example usage
# if __name__ == '__main__':
#     # Example graph
#     G = nx.Graph()
#     G.add_weighted_edges_from([
#         (0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 4), (0, 2, 3)
#     ])

#     # Degree constraints
#     degree_constraints = {0: 2, 1: 2, 2: 2, 3: 2}

#     # Initialize the DCMST problem
#     dcmst_qubo = DCMST_QUBO(G, degree_constraints, root=0)
#     dcmst_qubo.configure_variables()
#     dcmst_qubo.define_objective_function()
#     dcmst_qubo.add_constraints()

#     # Solve the QUBO (example solver can be plugged in here)
#     from qiskit_optimization.algorithms import MinimumEigenOptimizer
#     from qiskit_algorithms import QAOA
#     from qiskit.primitives import Sampler
#     from qiskit_aer import AerSimulator

#     sampler = Sampler(backend=AerSimulator())
#     qaoa = QAOA(sampler=sampler, reps=1)
#     optimizer = MinimumEigenOptimizer(qaoa)

#     result = dcmst_qubo.solve(optimizer)
#     print(result)
