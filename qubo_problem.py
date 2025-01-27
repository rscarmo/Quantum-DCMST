from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_algorithms import QAOA, SamplingVQE

# from qiskit.primitives import Sampler
from qiskit.primitives import BackendSampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKolkataV2
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer
from qiskit_aer import AerSimulator
from qiskit_optimization.problems.variable import VarType
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZGate, RXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile
from itertools import product
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.circuit.library import PhaseGate
from qiskit_algorithms import VQE
from qiskit.circuit.library import StatePreparation
from qiskit.synthesis.evolution import LieTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import QasmSimulator
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import minimize


import time
import networkx as nx
import numpy as np
import math
import copy
import pandas as pd
import sys

# import matplotlib
# matplotlib.use('Agg')


class DCMST_QUBO:
    def __init__(
        self,
        G,
        degree_constraints,
        config,
        mixer=None,
        initial_state=None,
        regularization=0,
        seed=42,
        redundancy=False,
        VQE=False,
        fake_backend = False
    ):
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
        self.root = 0
        self.config = config
        self.mixer = mixer
        self.initial_state = initial_state
        self.seed = seed
        self.VQE = VQE
        self.epsilon = regularization
        self.redundancy = redundancy
        self.fake_backend = fake_backend
        self.num_qubits = 0
        self.var_names = None

        self.objective_func_vals = []
        self.qaoa_circuit = None

        self.max_degree = self.degree_constraints.get(self.n - 1, self.n - 1)
        self.binary_bits = int(np.ceil(np.log2(self.max_degree + 1)))
        # Define the penalty coefficient
        num_vertices = self.n
        m = max(data["weight"] for _, _, data in self.G.edges(data=True))
        self.P_I = (num_vertices - 1) * m + 1
        print("P_I", self.P_I)

    def configure_variables(self):
        """Define variables for edge inclusion (e_{u,v}), order (x_{u,v}), and degree counters (z_{v,i})."""
        # Edge inclusion variables
        for u, v, w in self.G.edges(data="weight"):
            var_name_e = f"e_{u}_{v}"
            self.qubo.binary_var(name=var_name_e)

            # Only one direction for edges involving the root
            if u != self.root and v != self.root:
                var_name_e_reverse = f"e_{v}_{u}"
                self.qubo.binary_var(name=var_name_e_reverse)

        # Ordering variables x_{u,v}
        for u in range(1, self.n):
            if u == self.root:
                continue
            for v in range(u + 1, self.n):
                if v == self.root:
                    continue
                var_name_x = f"x_{u}_{v}"
                self.qubo.binary_var(name=var_name_x)

        # Degree counter variables z_{v,i}
        for v in self.G.nodes():
            for i in range(self.binary_bits):
                var_name_z = f"z_{v}_{i}"
                self.qubo.binary_var(name=var_name_z)

        self.var_names = self.qubo.variables

    def define_objective_function(self):
        """Define the objective function including penalties."""
        # Initialize dictionaries to accumulate terms
        linear_terms = {}
        quadratic_terms = {}

        # 1. Minimize the total weight of selected edges
        # Root edges: Only consider one direction (v0 -> u)
        for u in self.G.neighbors(self.root):
            edge_var = f"e_{self.root}_{u}"
            weight = self.G[self.root][u]["weight"]
            linear_terms[edge_var] = linear_terms.get(edge_var, 0) + weight

        # Non-root edges: Include both directions (u -> v and v -> u)
        for u, v, w in self.G.edges(data="weight"):
            if u != self.root and v != self.root:
                edge_var_uv = f"e_{u}_{v}"
                edge_var_vu = f"e_{v}_{u}"
                linear_terms[edge_var_uv] = linear_terms.get(edge_var_uv, 0) + w
                linear_terms[edge_var_vu] = linear_terms.get(edge_var_vu, 0) + w

        # 2. Add penalties for constraints
        # Acyclicity via Topological Ordering Penalty Terms
        for u in range(1, self.n):
            for v in range(u + 1, self.n):
                for w in range(v + 1, self.n):
                    if u == self.root or v == self.root or w == self.root:
                        continue

                    # Define variable names
                    var_name_x_uv = f"x_{u}_{v}"
                    var_name_x_vw = f"x_{v}_{w}"
                    var_name_x_uw = f"x_{u}_{w}"

                    # Ensure all required variables exist
                    variables = [var_name_x_uv, var_name_x_vw, var_name_x_uw]
                    if not all(var in self.qubo.variables_index for var in variables):
                        continue

                    # Accumulate linear term: +P_I * x_{u,w}
                    linear_terms[var_name_x_uw] = (
                        linear_terms.get(var_name_x_uw, 0) + self.P_I
                    )

                    # Accumulate quadratic terms
                    quadratic_terms[tuple(sorted([var_name_x_uv, var_name_x_vw]))] = (
                        quadratic_terms.get(
                            tuple(sorted([var_name_x_uv, var_name_x_vw])), 0
                        )
                        + self.P_I
                    )
                    quadratic_terms[tuple(sorted([var_name_x_uv, var_name_x_uw]))] = (
                        quadratic_terms.get(
                            tuple(sorted([var_name_x_uv, var_name_x_uw])), 0
                        )
                        - self.P_I
                    )
                    quadratic_terms[tuple(sorted([var_name_x_uw, var_name_x_vw]))] = (
                        quadratic_terms.get(
                            tuple(sorted([var_name_x_uw, var_name_x_vw])), 0
                        )
                        - self.P_I
                    )

        # Alignment Constraint (Term (ii))
        for u, v in self.G.edges():
            if u < v and u != self.root and v != self.root:
                var_name_e_uv = f"e_{u}_{v}"
                var_name_e_vu = f"e_{v}_{u}"
                var_name_x_uv = f"x_{u}_{v}"

                # Ensure all required variables exist
                variables = [var_name_e_uv, var_name_e_vu, var_name_x_uv]
                if not all(var in self.qubo.variables_index for var in variables):
                    continue

                # Accumulate linear term: +P_I * e_{u,v}
                linear_terms[var_name_e_uv] = (
                    linear_terms.get(var_name_e_uv, 0) + self.P_I
                )

                # Accumulate quadratic terms
                quadratic_terms[tuple(sorted([var_name_e_uv, var_name_x_uv]))] = (
                    quadratic_terms.get(
                        tuple(sorted([var_name_e_uv, var_name_x_uv])), 0
                    )
                    - self.P_I
                )
                quadratic_terms[tuple(sorted([var_name_e_vu, var_name_x_uv]))] = (
                    quadratic_terms.get(
                        tuple(sorted([var_name_e_vu, var_name_x_uv])), 0
                    )
                    + self.P_I
                )

        # 3. Set the final objective function in the QUBO
        self.qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

    def add_constraints(self):
        """Add constraints to enforce the DCMST problem conditions."""
        # 1.1 and 1.2 are redundant restrictions, trying to improve the results.
        # ------------------------------------------------------------------------------------
        if self.redundancy:
            # 1.1 Edge count constraint: sum of selected edges = n - 1
            edge_count_constraint = {
                f"e_{u}_{v}": 1 for u, v in self.G.edges() if v != self.root
            }
            self.qubo.linear_constraint(
                linear=edge_count_constraint,
                sense="==",
                rhs=self.n - 1,
                name="edge_count_constraint",
            )

            # 1.2 Adicionar restrições para garantir e_{u,v} + e_{v,u} <= 1, quando não envolvem a raíz
            for u, v in self.G.edges:
                if u != self.root and v != self.root:
                    # Evitar redundância (adicionar apenas uma vez por par de arestas)
                    self.qubo.linear_constraint(
                        linear={f"e_{u}_{v}": 1, f"e_{v}_{u}": 1},
                        sense="<=",
                        rhs=1,  # Somente uma dessas variáveis pode ser 1 ao mesmo tempo
                        name=f"anti_symmetric_constraint_{u}_{v}",
                    )
        # --------------------------------------------------------------------------------------

        # 2. Acyclicity constraints using ordering variables x_{u,v}
        # These are now handled in define_penalty_terms()

        # 3. Edge alignment constraints: ensure consistency between e_{u,v} and x_{u,v}
        # These are now handled in define_penalty_terms()

        """Add connectivity constraints (Constraint iii) as linear constraints."""
        for v in self.G.nodes():
            if v == self.root:
                continue  # Skip the root node

            # Sum of incoming edges to v
            incoming_edges = {f"e_{u}_{v}": 1 for u in self.G.neighbors(v)}

            # Add linear constraint: sum of incoming edges = 1
            self.qubo.linear_constraint(
                linear=incoming_edges,
                sense="==",
                rhs=1,
                name=f"connectivity_constraint_{v}",
            )

        """Add degree constraints (Constraint iv) as linear constraints."""
        for v in self.G.nodes():
            # Get the maximum allowed degree for v (or default)
            # max_degree = self.degree_constraints.get(v, self.n - 1)

            # Number of standard binary bits

            # -----------------------------
            #  1) Construct z_v as:
            #     z_v = sum_{i=0}^{k-1} 2^i z_{v,i} + (Delta+1 - 2^k)*z_{v,k}
            # -----------------------------
            all_terms = {}

            # (A) Standard bits (0..k-1)
            for i in range(self.binary_bits - 1):
                bit_name = f"z_{v}_{i}"
                # Coefficient +2^i
                all_terms[bit_name] = all_terms.get(bit_name, 0.0) + (2**i)

            # (B) Special last bit (kth bit):
            #     (Delta+1 - 2^k) * z_{v,k}
            special_bit_name = f"z_{v}_{self.binary_bits-1}"
            all_terms[special_bit_name] = all_terms.get(special_bit_name, 0.0) + (
                (self.max_degree + 1) - 2 ** (self.binary_bits - 1)
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
                        all_terms[edge_out_name] = (
                            all_terms.get(edge_out_name, 0.0) - 1.0
                        )

            # -----------------------------
            #  3) Enforce equality: z_v - sum_of_edges = 0
            # -----------------------------
            self.qubo.linear_constraint(
                linear=all_terms, sense="==", rhs=0.0, name=f"degree_constraint_{v}"
            )

    def brute_force_solution(self):
        """
        Exhaustively enumerates all 2^N possible solutions to the QUBO,
        checks feasibility for each, and returns the best (lowest cost)
        feasible bitstring and its cost.

        Returns:
            best_bitstring (str): A string of '0's and '1's representing the best found solution.
            best_cost (float): The objective function value of that best solution.
        """
        from itertools import product
        import math

        num_vars = len(self.qubo.variables)
        best_cost = math.inf
        best_bitstring = None

        # Loop over all 2^N possible assignments
        for bits_tuple in product([0, 1], repeat=num_vars):
            # Convert tuple -> list[float] for Qiskit
            candidate_x = list(map(float, bits_tuple))

            # Check feasibility (i.e., constraints) within some numerical tolerance
            if self.qubo.is_feasible(candidate_x):
                # Evaluate the objective value if feasible
                cost = self.qubo.objective.evaluate(candidate_x)
                # Track the best feasible solution
                if cost < best_cost:
                    best_cost = cost
                    best_bitstring = bits_tuple  # keep the raw tuple

        # Convert the best_bitstring tuple to a single string
        if best_bitstring is not None:
            best_bitstring_str = ''.join(map(str, best_bitstring))
        else:
            best_bitstring_str = None

        return best_bitstring_str, best_cost



    def configure_backend(self):
        if self.config.SIMULATION == "True":
            if not self.fake_backend:
                print("Proceeding with simulation...")
                backend = AerSimulator()
            else:
                print("Proceeding with simulation in Fake IBM_Brisbane using AerSimulator...")
                service = QiskitRuntimeService(
                    channel="ibm_quantum", token=self.config.QXToken
                )                
                real_backend = service.backend("ibm_brisbane")
                
                backend = AerSimulator.from_backend(real_backend)
            # backend = QasmSimulator()
            backend.set_options(seed_simulator=self.seed)
        else:
            print("Proceeding with IBM Quantum hardware...")
            service = QiskitRuntimeService(
                channel="ibm_quantum", token=self.config.QXToken
            )
            # backend = service.least_busy(min_num_qubits=127, operational=True, simulator=False)
            backend = service.backend("ibm_brisbane")
            print(f"Connected to {backend.name}!")
        return backend

    def relax_problem(self, problem) -> QuadraticProgram:
        """Change all variables to continuous."""
        relaxed_problem = copy.deepcopy(problem)
        for variable in relaxed_problem.variables:
            if "x" not in variable.name:
                variable.vartype = VarType.CONTINUOUS

        return relaxed_problem

    def compute_theta(self, c_star):
        """
        Compute the theta value based on c_star and epsilon.

        Parameters:
        - c_star: The c_i^* value for a specific variable.
        - epsilon: The threshold value (default is 0.25).

        Returns:
        - theta: The computed theta value.
        """
        if self.epsilon <= c_star <= 1 - self.epsilon:
            theta = 2 * np.arcsin(np.sqrt(c_star))
        elif c_star < self.epsilon:
            theta = 2 * np.arcsin(np.sqrt(self.epsilon))
        else:  # c_star > 1 - self.epsilon
            theta = 2 * np.arcsin(np.sqrt(1 - self.epsilon))
        return theta

    def initial_state_RY(self, thetas):
        init_qc = QuantumCircuit(len(self.qubo.variables))
        for idx, theta in enumerate(thetas):
            init_qc.ry(theta, idx)

        init_qc.draw(output="mpl", style="clifford")

        return init_qc

    def mixer_warm(self, thetas):
        beta = Parameter("β")

        ws_mixer = QuantumCircuit(len(self.qubo.variables))
        for idx, theta in enumerate(thetas):
            ws_mixer.ry(-theta, idx)
            ws_mixer.rz(-2 * beta, idx)
            ws_mixer.ry(theta, idx)

        ws_mixer.draw(output="mpl", style="clifford")

        return ws_mixer

    def has_cycle(self, adjacency_list, nodes):
        """
        Detects if there's a cycle in the directed graph represented by adjacency_list.

        Parameters:
            adjacency_list (dict): Adjacency list of the graph.
            nodes (list): List of nodes to consider in the graph.

        Returns:
            bool: True if a cycle is detected, False otherwise.
        """
        visited = {node: False for node in nodes}
        rec_stack = {node: False for node in nodes}

        def dfs(node):
            visited[node] = True
            rec_stack[node] = True

            for neighbor in adjacency_list.get(node, []):
                if not visited[neighbor]:
                    if dfs(neighbor):
                        return True
                elif rec_stack[neighbor]:
                    return True  # Cycle detected

            rec_stack[node] = False
            return False

        for node in nodes:
            if not visited[node]:
                if dfs(node):
                    return True
        return False

    def get_valid_bitstrings(self, n_non_root):
        """
        Generates all valid bitstring configurations for a given number of non-root nodes,
        ensuring that no cycles are formed.

        Parameters:
            n_non_root (int): Number of non-root nodes in the graph.

        Returns:
            List[str]: List of valid bitstrings representing acyclic configurations.
        """
        valid_bitstrings = []

        # Generate all unique ordered pairs (u, v) where u < v
        variable_pairs = []
        for u in range(1, n_non_root + 1):
            for v in range(u + 1, n_non_root + 1):
                variable_pairs.append((u, v))

        num_vars = len(variable_pairs)

        # Iterate over all possible bitstrings
        for bits in product([0, 1], repeat=num_vars):
            adjacency_list = {node: [] for node in range(1, n_non_root + 1)}

            # Map bits to directed edges
            for idx, bit in enumerate(bits):
                u, v = variable_pairs[idx]
                if bit == 1:
                    adjacency_list[u].append(v)  # u precedes v
                else:
                    adjacency_list[v].append(u)  # v precedes u

            # Check for cycles
            if not self.has_cycle(adjacency_list, list(range(1, n_non_root + 1))):
                # Convert bits tuple to string
                bitstring = "".join(map(str, bits))
                valid_bitstrings.append(bitstring)

        return valid_bitstrings

    def generate_z_states(self):
        """
        Gera a lista de estados binários permitidos para as variáveis z_v,i,
        baseando-se no valor de delta.

        Retorna:
        List[str]: Lista de estados binários permitidos.
        """
        # Número de bits necessários para representar z_v
        n_bits = (self.max_degree).bit_length()  # ceil(log2(delta + 1))

        # Gera todos os estados possíveis com n_bits, mas limitados de 0 a delta
        all_states = [
            format(
                i, f"0{n_bits}b"
            )  # Formata o número como uma string binária de n_bits
            for i in range(self.max_degree + 1)  # Inclui apenas valores de 0 a delta
        ]

        return all_states

    def generate_states_excluding_specific(self, n_qubits, exclude_state):
        """
        Gera todos os estados possíveis para n_qubits, excluindo um estado específico.

        Parâmetros:
        n_qubits (int): Número de qubits.
        exclude_state (str): O estado binário a ser excluído (e.g., '000').

        Retorna:
        List[str]: Lista de estados possíveis em formato binário, exceto o estado excluído.
        """
        # Gera todos os estados binários possíveis
        all_states = product([0, 1], repeat=n_qubits)

        # Exclui o estado especificado
        valid_states = [
            "".join(map(str, state))
            for state in all_states
            if "".join(map(str, state)) != exclude_state
        ]

        return valid_states

    def prepare_superposition(self, qc, valid_states, qubit_indices):
        """
        Prepara uma superposição uniforme dos estados válidos usando Qiskit.

        Parâmetros:
        qc (QuantumCircuit): O circuito quântico onde a superposição será preparada.
        valid_states (list of str): Lista de estados válidos no formato binário (e.g., ["000", "001"]).
        qubit_indices (list of int): Índices dos qubits no circuito onde os estados serão aplicados.

        Retorna:
        QuantumCircuit: O circuito com a superposição preparada.
        """
        # Número de qubits
        n_qubits = len(qubit_indices)
        # Número de estados válidos
        n_states = len(valid_states)

        # Calcula os amplitudes necessários para a superposição uniforme
        amplitude = 1 / np.sqrt(n_states)
        state_vector = np.zeros(2**n_qubits, dtype=complex)

        # Define as amplitudes para os estados válidos
        for state in valid_states:
            index = int(state, 2)  # Converte o estado binário para um índice decimal
            state_vector[index] = amplitude

        # Adiciona o estado inicial ao circuito
        qc.initialize(state_vector, qubit_indices)

        return qc

    def initial_state_OHE(self):
        init_qc = QuantumCircuit(len(self.qubo.variables))

        idx = 0
        self.group_e_v0 = []
        group_e_uv = []
        self.groups_e_uv = []
        group_z_i = []
        self.groups_z_i = []
        self.group_x = []

        # (e_uv = 1 and e_vu = 0) or (e_uv = 0 and e_vu = 1). So, the state |11> is forbidden
        self.valid_e_uv_states = self.generate_states_excluding_specific(2, "11")
        # self.valid_z_i_states = self.generate_z_states()

        for variable in self.qubo.variables:

            if "e_" + str(self.root) in variable.name:
                self.group_e_v0.append(idx)
            elif "e" in variable.name:
                if idx not in group_e_uv:
                    group_e_uv.append(idx)
                    group_e_uv.append(idx + 1)
                    init_qc = self.prepare_superposition(
                        init_qc, self.valid_e_uv_states, group_e_uv
                    )
                    self.groups_e_uv.append(group_e_uv)
                else:
                    group_e_uv = []

            elif "z" in variable.name:
                if idx not in group_z_i:
                    for i in range(self.binary_bits):
                        group_z_i.append(idx + i)
                    self.groups_z_i.append(group_z_i)
                    # init_qc = self.prepare_superposition(init_qc, self.valid_z_i_states, group_z_i)
                else:
                    group_z_i = []
            elif "x" in variable.name:
                self.group_x.append(idx)

            idx += 1

        self._num_x = len(self.group_x)
        self._num_e_v0 = len(self.group_e_v0)
        # I will use this valid_x_states list, ['000000', '000001', ...], to prepare the initial state
        self.valid_x_states = self.get_valid_bitstrings(self.n - 1)
        print(self.valid_x_states)
        init_qc = self.prepare_superposition(init_qc, self.valid_x_states, self.group_x)

        # I'm not certain if this worth the cost
        # valid_e_v0_states = self.generate_states_excluding_specific(self._num_e_v0, '0'*self._num_e_v0)
        # self.prepare_superposition(init_qc, valid_e_v0_states, group_e_v0)
        for id in self.group_e_v0:
            init_qc.h(id)

        for ids in self.groups_z_i:
            for id in ids:
                init_qc.h(id)

        # init_qc.draw(output="mpl", style="clifford")
        init_qc.draw(output="mpl").savefig("init_qc.png")

        return init_qc

    def mixer_LogicalX(self, hamiltonian, qubit_indices, beta, circuit):
        """
        Constructs and appends the PauliEvolutionGate for the mixer Hamiltonian.

        Args:
            hamiltonian (list of tuples): Each tuple contains a main Pauli string and a list of constraint tuples.
                                          Example: [('IIX', [('III', 1.0), ('IZI', 1.0)]), ('IXX', [('IIZ', 1.0), ('IZZ', 1.0)]), ('XII', [('III', 1.0), ('IZZ', 1.0)])]
            qubit_indices (list of int): Qubit indices the Pauli strings act upon.
            beta (float): Parameter for the mixer.
            circuit (QuantumCircuit): The quantum circuit to append the mixer to.

        Returns:
            QuantumCircuit: The updated circuit with the mixer appended.
        """

        def parse_pauli_label(pauli_label):
            """
            Parses a Pauli label string with a phase prefix and returns the phase and operator string.

            Args:
                pauli_label (str): The Pauli label, e.g., '-iZY', 'IX', 'iXZ', '-II'

            Returns:
                tuple: (phase, operator_str)
                    phase: complex number (1, -1, 1j, -1j)
                    operator_str: str, the Pauli operator string
            """
            if pauli_label.startswith("-i"):
                phase = -1j
                operator_str = pauli_label[2:]
            elif pauli_label.startswith("i"):
                phase = 1j
                operator_str = pauli_label[1:]
            elif pauli_label.startswith("-"):
                phase = -1.0
                operator_str = pauli_label[1:]
            elif pauli_label.startswith("+"):
                phase = 1.0
                operator_str = pauli_label[1:]
            else:
                phase = 1.0
                operator_str = pauli_label
            return (phase, operator_str)

        num_qubits = len(qubit_indices)
        # Initialize lists for Pauli strings and coefficients
        pauli_strings = []
        coefficients = []

        for main_pauli_str, constraints in hamiltonian:
            main_pauli = Pauli(main_pauli_str)
            for constraint_pauli_str, coef in constraints:
                constraint_pauli = Pauli(constraint_pauli_str)
                # Compute the product Pauli and phase
                combined_pauli = main_pauli.dot(constraint_pauli)
                # Convert Pauli object back to string
                combined_pauli_str = combined_pauli.to_label()
                print(f"AQUI ESTÁ O OPERADOR: {combined_pauli_str}")
                # Parse to get phase and operator
                phase, operator_str = parse_pauli_label(combined_pauli_str)
                print(f"    Parsed Phase: {phase}, Operator: {operator_str}")
                # Adjust coefficient by phase
                adjusted_coef = coef * phase
                # If the coefficient is zero or Pauli is all 'I's, skip
                if adjusted_coef == 0.0 or operator_str == "I" * num_qubits:
                    print(
                        f"    Skipping term: {operator_str} with adjusted coefficient: {adjusted_coef}"
                    )
                    continue  # Avoid adding 'III' to the Hamiltonian

                # Append to lists
                pauli_strings.append(operator_str)
                coefficients.append(adjusted_coef.real)  # Use the real part
                print(
                    f"    Added Pauli string: {operator_str} with coefficient: {adjusted_coef.real}"
                )

        # Create SparsePauliOp with the list
        sp_op_total = SparsePauliOp.from_list(list(zip(pauli_strings, coefficients)))

        # Verify the constructed Hamiltonian
        print("Constructed Mixer Hamiltonian:")
        print(sp_op_total)

        # Create Pauli Evolution Gate using LieTrotter decomposition
        trotter = LieTrotter(reps=1)
        evolution_gate = PauliEvolutionGate(sp_op_total, time=beta, synthesis=trotter)

        # Append the gate to the circuit acting on qubit_indices
        circuit.append(evolution_gate, qubit_indices)

        return circuit

    def mixer_customized(self):
        mixer_circuit = QuantumCircuit(len(self.qubo.variables))
        beta = Parameter("β")

        # For the e_v0_u variables
        for id in self.group_e_v0:
            mixer_circuit.rx(-2 * beta, id)

        # For the e_uv and e_vu variables
        # Needs to search in the hamiltonian_mixers.cvs using the valid_states
        df = pd.read_csv("hamiltonian_mixers.csv")

        for ids in self.groups_e_uv:
            # Procurar o Hamiltoniano correspondente ao estado
            filtered_row = df[df["states"] == str(self.valid_e_uv_states)]
            if not filtered_row.empty:
                hamiltonian = eval(filtered_row.iloc[0]["hamiltonian"])
                print(hamiltonian, ids)
                mixer_circuit = self.mixer_LogicalX(
                    hamiltonian, ids, beta, mixer_circuit
                )

            else:
                print(
                    f"Estado {self.valid_e_uv_states} referente às variaveis e_uv não encontrado no arquivo CSV."
                )
                sys.exit()

        # For the x_uv
        # Needs to search in the hamiltonian_mixers.cvs using the valid_states
        for ids in self.group_x:
            mixer_circuit.rx(-2 * beta, ids)

        # filtered_row = df[df['states'] == str(self.valid_x_states)]
        # if len(self.group_x) == 1:
        #     mixer_circuit.rx(-2*beta, self.group_x)
        # else:
        #     if not filtered_row.empty:  # Verificar se encontrou o estado
        #         hamiltonian = eval(filtered_row.iloc[0]['hamiltonian'])  # Avaliar o Hamiltoniano como objeto Python
        #         print(hamiltonian, self.group_x)

        #         # Chamar a função mixer_LogicalX com os dados encontrados
        #         mixer_circuit = self.mixer_LogicalX(hamiltonian, self.group_x, beta, mixer_circuit)

        #     else:
        #         print(f"Estado {self.valid_x_states} referente às variaveis x_uv não encontrado no arquivo CSV.")
        #         sys.exit()

        # For the z_vi
        # Needs to search in the hamiltonian_mixers.cvs using the valid_states
        for ids in self.groups_z_i:
            mixer_circuit.rx(-2 * beta, ids)
            # filtered_row = df[df['states'] == str(self.valid_z_i_states)]
            # print('AQUI ESTÃO OS ESTADOS VÁLIDOS DE Z:', self.valid_z_i_states)
            # if not filtered_row.empty:  # Verificar se encontrou o estado
            #     hamiltonian = eval(filtered_row.iloc[0]['hamiltonian'])  # Avaliar o Hamiltoniano como objeto Python
            #     print(hamiltonian, ids)

            #     # Chamar a função mixer_LogicalX com os dados encontrados
            #     mixer_circuit = self.mixer_LogicalX(hamiltonian, ids, beta, mixer_circuit)
            # else:
            #     print(f"Estado {self.valid_x_states} referente às variaveis z_vi não encontrado no arquivo CSV.")
            #     sys.exit()

        return mixer_circuit

    def cost_func_estimator(self, params, ansatz, hamiltonian, estimator, offset=0.0):

        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs

        self.objective_func_vals.append(cost)

        return cost + offset
    
    def evaluate_bitstring_cost(self, bitstring: str) -> float:
        """
        Given a candidate bitstring (e.g. '010101'),
        returns the objective value of that bitstring for this QUBO.
        
        Parameters:
            bitstring (str): A string of '0'/'1' characters whose length 
                            must match the total number of variables in self.qubo.

        Returns:
            float: The cost (objective value) evaluated at the bitstring.
        """
        # Convert each character in the bitstring to float 0.0 or 1.0
        x = [float(b) for b in bitstring]

        # Safety check: ensure bitstring length matches the number of QUBO variables
        if len(x) != len(self.qubo.variables):
            raise ValueError(
                f"Bitstring length ({str(x)}) does not match the expected number "
                f"of variables ({len(self.qubo.variables)}) in the QuadraticProgram."
            )

        # Evaluate the (unconstrained) objective function for this bitstring
        converter = QuadraticProgramToQubo(penalty=self.P_I)
        qubo_with_penalties = converter.convert(self.qubo)

        cost = qubo_with_penalties.objective.evaluate(x)
        return cost

    
    def time_execution_feasibility(self, backend):
        # Retrieve backend properties
        properties = backend.properties()

        # Extract gate durations
        gate_durations = {}
        for gate in properties.gates:
            gate_name = gate.gate
            if gate.parameters:
                duration = gate.parameters[0].value  # Duration in seconds
                gate_durations[gate_name] = duration

        print("Gate durations (in seconds):")
        for gate, duration in gate_durations.items():
            print(f"{gate}: {duration * 1e9:.2f} ns")

        # Calculate total execution time
        total_time = 0
        for instruction, qargs, cargs in self.qaoa_circuit.data:
            gate_name = instruction.name
            gate_time = gate_durations.get(gate_name, 0)
            total_time += gate_time

        print(f"Total circuit execution time: {total_time * 1e6:.2f} µs")

        # Extract coherence times with qubit indices
        coherence_times = {}
        for qubit_index, qubit in enumerate(properties.qubits):
            T1 = None
            T2 = None
            for param in qubit:
                if param.name == 'T1':
                    T1 = param.value
                elif param.name == 'T2':
                    T2 = param.value
            coherence_times[qubit_index] = {'T1': T1, 'T2': T2}
            print(f"Qubit {qubit_index}: T1 = {T1*1e6:.2f} µs, T2 = {T2*1e6:.2f} µs")

        # Access the layout to map virtual qubits to physical qubits
        transpile_layout = self.qaoa_circuit._layout  # Note the underscore before 'layout'

        layout = transpile_layout.final_layout
        
        # Retrieve the virtual-to-physical qubit mapping
        virtual_to_physical = layout.get_virtual_bits()

        # Determine which physical qubits are used in the circuit
        used_physical_qubits = set(virtual_to_physical.values())

        # Now, get the minimum T1 and T2 among the used physical qubits
        min_T1 = min(coherence_times[q_index]['T1'] for q_index in used_physical_qubits)
        min_T2 = min(coherence_times[q_index]['T2'] for q_index in used_physical_qubits)

        # Compare execution time to thresholds
        threshold_T1 = 0.1 * min_T1
        threshold_T2 = 0.1 * min_T2

        print(f"Thresholds: 10% T1 = {threshold_T1*1e6:.2f} µs, 10% T2 = {threshold_T2*1e6:.2f} µs")
        print(f"Circuit execution time: {total_time*1e6:.2f} µs")

        if total_time < threshold_T1 and total_time < threshold_T2:
            print("Execution time is within acceptable limits.")
        else:
            print("Execution time may be too long; consider optimizing your circuit.")       

    def prepare_metaloss(self, optimizer, p=1, parameters=None):
        """
        Retorna função de custo para o otimizador metaheuristic
        """

        # Convert the problem with constraints into an unconstrained QUBO
        converter = QuadraticProgramToQubo(penalty=self.P_I)
        qubo = converter.convert(self.qubo)

        # Print the Ising model
        print(qubo.to_ising())

        qubo_ops, offset = qubo.to_ising()

        backend = self.configure_backend()
        sampler = BackendSampler(backend=backend)

        # Define a callback function to track progress
        def callback(params):
            print(f"Current parameters: {params}")

        # Set up QAOA with the callback

        np.random.seed(self.seed)

        # Define the seed that will be used in the optimization process
        algorithm_globals.random_seed = self.seed

        # Generate initial parameters using the seed
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * p)
        if not self.VQE:
            if self.mixer == "Warm":
                print(self.qubo.prettyprint())
                qp = self.relax_problem(qubo)
                print(qp.prettyprint())
                sol = CplexOptimizer().solve(qp)
                print(sol.prettyprint())
                c_stars = sol.samples[0].x

                # Example usage:
                thetas = [self.compute_theta(c_star) for c_star in c_stars]

                print(thetas)

                self.mixer = self.mixer_warm(thetas)
                self.initial_state = self.initial_state_RY(thetas)
                self.mixer.draw(output="mpl").savefig("mixer_circuit.png")
            elif self.mixer == "LogicalX":
                self.initial_state = self.initial_state_OHE()
                self.mixer = self.mixer_customized()
                self.mixer.draw(output="mpl").savefig("mixer_circuit.png")

            qaoa_mes = QAOAAnsatz(
                cost_operator=qubo_ops,
                reps=p,
                mixer_operator=self.mixer,
                initial_state=self.initial_state,
            )
            qaoa_mes.measure_all()

            # Create a custom pass manager
            pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

            # Transpile the circuit
            self.qaoa_circuit = pm.run(qaoa_mes)

            if self.fake_backend or self.config.SIMULATION == False:
                try:
                    self.time_execution_feasibility(backend)
                except:
                    pass

            estimator = Estimator(backend)
            estimator.options.default_shots = 1000

            if self.config.SIMULATION == False:
                # Set simple error suppression/mitigation options
                estimator.options.dynamical_decoupling.enable = True
                estimator.options.dynamical_decoupling.sequence_type = "XY4"
                estimator.options.twirling.enable_gates = True
                estimator.options.twirling.num_randomizations = "auto"

        else:
            # Build the parameterized ansatz for VQE
            from qiskit.circuit.library import TwoLocal

            ansatz = TwoLocal(
                len(self.qubo.variables),
                rotation_blocks="ry",
                entanglement_blocks="cx",
                entanglement="linear",
                reps=p,
            )
            initial_parameters = np.random.random(ansatz.num_parameters)

            sampling_vqe = SamplingVQE(
                sampler=sampler,
                ansatz=ansatz,
                optimizer=optimizer,
                initial_point=initial_parameters,
                callback=callback,  # if you have a callback function defined
            )

            start_time = time.time()
            result = sampling_vqe.compute_minimum_eigenvalue(operator=qubo_ops)
            end_time = time.time()

            self.execution_time = end_time - start_time
            print(f"VQE ground state energy (no offset): {result.eigenvalue.real:.5f}")
            print(
                f"VQE ground state energy (with offset): {(result.eigenvalue + offset).real:.5f}"
            )
            print(f"Execution time: {self.execution_time:.4f} seconds")

            # Retrieve the optimal parameters
            optimal_params = result.optimal_point
            print("Optimal parameters:", optimal_params)

            # Bind the optimal parameters to the ansatz
            bound_ansatz = ansatz.assign_parameters(optimal_params)

            # Sample the circuit using the sampler
            sampled_result = sampler.run(bound_ansatz, shots=1024).result()
            quasi_dist = sampled_result.quasi_dists[0]  # Assuming a single circuit

            shots = 1024
            float_counts = quasi_dist.binary_probabilities()
            counts = {
                state: int(round(prob * shots)) for state, prob in float_counts.items()
            }

        # Return or store them
        print(counts)
        # return counts

        # qaoa_result = minimize(
        #     self.cost_func_estimator,
        #     initial_params,
        #     args=(self.qaoa_circuit, qubo_ops, estimator, offset),
        #     method="COBYLA",
        #     tol=1e-2,
        #     callback=callback
        # )

        def _fn(params):
            params = np.array(params).flatten()
            l = self.cost_func_estimator(
                params, self.qaoa_circuit, qubo_ops, estimator, offset
            )
            # print(f"Cost: {l}")
            return l
        return _fn

    def solve_problem(self, optimizer, p=1, parameters=None):
        # Convert the problem with constraints into an unconstrained QUBO
        converter = QuadraticProgramToQubo(penalty=self.P_I)
        qubo = converter.convert(self.qubo)

        # Print the Ising model
        print(qubo.to_ising())

        qubo_ops, offset = qubo.to_ising()

        backend = self.configure_backend()

        # Define a callback function to track progress
        def callback(params):
            print(f"Current parameters: {params}")

        # Set up QAOA with the callback

        np.random.seed(self.seed)

        # Define the seed that will be used in the optimization process
        algorithm_globals.random_seed = self.seed

        # Generate initial parameters using the seed
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * p)
        if not self.VQE:
            if self.mixer == "Warm":
                print(self.qubo.prettyprint())
                qp = self.relax_problem(qubo)
                print(qp.prettyprint())
                sol = CplexOptimizer().solve(qp)
                print(sol.prettyprint())
                c_stars = sol.samples[0].x

                # Example usage:
                thetas = [self.compute_theta(c_star) for c_star in c_stars]

                print(thetas)

                self.mixer = self.mixer_warm(thetas)
                self.initial_state = self.initial_state_RY(thetas)
                self.mixer.draw(output="mpl").savefig("mixer_circuit.png")

            elif self.mixer == "LogicalX":
                self.initial_state = self.initial_state_OHE()
                self.mixer = self.mixer_customized()
                self.mixer.draw(output="mpl").savefig("mixer_circuit.png")

            qaoa_mes = QAOAAnsatz(
                cost_operator=qubo_ops,
                reps=p,
                mixer_operator=self.mixer,
                initial_state=self.initial_state,
            )
            qaoa_mes.measure_all()

            # Create a custom pass manager
            pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

            # Transpile the circuit
            self.qaoa_circuit = pm.run(qaoa_mes)

            if self.fake_backend or self.config.SIMULATION == False:
                try:
                    self.time_execution_feasibility(backend)
                except:
                    pass

            estimator = Estimator(backend)
            estimator.options.default_shots = 1000

            if self.config.SIMULATION == False:
                # Set simple error suppression/mitigation options
                estimator.options.dynamical_decoupling.enable = True
                estimator.options.dynamical_decoupling.sequence_type = "XY4"
                estimator.options.twirling.enable_gates = True
                estimator.options.twirling.num_randomizations = "auto"
        else:
            # Build the parameterized ansatz for VQE
            from qiskit.circuit.library import TwoLocal

            sampler = BackendSampler(backend=backend)            

            ansatz = TwoLocal(
                len(self.qubo.variables),
                rotation_blocks="ry",
                entanglement_blocks="cx",
                entanglement="linear",
                reps=p,
            )
            initial_parameters = np.random.random(ansatz.num_parameters)

            sampling_vqe = SamplingVQE(
                sampler=sampler,
                ansatz=ansatz,
                optimizer=optimizer,
                initial_point=initial_parameters,
                callback=callback,  # if you have a callback function defined
            )

            start_time = time.time()
            result = sampling_vqe.compute_minimum_eigenvalue(operator=qubo_ops)
            end_time = time.time()

            self.execution_time = end_time - start_time
            print(f"VQE ground state energy (no offset): {result.eigenvalue.real:.5f}")
            print(
                f"VQE ground state energy (with offset): {(result.eigenvalue + offset).real:.5f}"
            )
            print(f"Execution time: {self.execution_time:.4f} seconds")

            # Retrieve the optimal parameters
            optimal_params = result.optimal_point
            print("Optimal parameters:", optimal_params)

            # Bind the optimal parameters to the ansatz
            bound_ansatz = ansatz.assign_parameters(optimal_params)

            # Sample the circuit using the sampler
            sampled_result = sampler.run(bound_ansatz, shots=1024).result()
            quasi_dist = sampled_result.quasi_dists[0]  # Assuming a single circuit

            shots = 1024
            float_counts = quasi_dist.binary_probabilities()
            counts = {
                state: int(round(prob * shots)) for state, prob in float_counts.items()
            }

            # Return or store them
            print(counts)
            return counts

        start_time = time.time()
        # qaoa_result = qaoa.solve(self.qubo)
        qaoa_result = minimize(
            self.cost_func_estimator,
            initial_params,
            args=(self.qaoa_circuit, qubo_ops, estimator, offset),
            method="COBYLA",
            tol=1e-2,
            callback=callback,
        )
        end_time = time.time()
        print(qaoa_result)

        self.execution_time = end_time - start_time
        # self.solution = qaoa_result.variables_dict
        print(f"Execution time: {self.execution_time} seconds")

        return qaoa_result.x

    def qubo_sample(self, optimal_params):
        backend = self.configure_backend()
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = 1000
        optimized_circuit = self.qaoa_circuit.assign_parameters(optimal_params)

        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=int(1e4))
        counts_int = job.result()[0].data.meas.get_int_counts()
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())

        # Reverse the bits of all keys in counts_bin
        reversed_distribution_bin = {
            key[::-1]: val / shots for key, val in counts_bin.items()
        }

        print(reversed_distribution_bin)

        return reversed_distribution_bin

    def print_number_of_qubits(self):
        """
        Calculate and print the number of qubits used in the problem.
        This is determined by the total number of binary variables in the QUBO.
        """
        self.num_qubits = len(self.qubo.variables)
        print(f"Number of qubits required: {self.num_qubits}")
