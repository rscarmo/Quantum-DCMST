from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit.primitives import Sampler
from qiskit.primitives import BackendSampler    
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
    def __init__(self, G, degree_constraints, config, root=0, mixer = None, initial_state = None,  
                 regularization = 0, seed = 42, warm_start = False, redundancy = False, VQE = False, Metaheuristic =False):
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
        self.warm_start = warm_start 
        self.VQE = VQE
        self.epsilon =  regularization
        self.redundancy = redundancy
        self.Metaheuristic = Metaheuristic
        self.num_qubits = 0

        self.max_degree = self.degree_constraints.get(self.n - 1, self.n - 1)        
        self.binary_bits = int(np.ceil(np.log2(self.max_degree+1)))        
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
        for v in self.G.nodes():
            for i in range(self.binary_bits):
                var_name_z = f'z_{v}_{i}'
                self.qubo.binary_var(name=var_name_z)

    def define_objective_function(self):
        """Define the objective function including penalties."""
        # Initialize dictionaries to accumulate terms
        linear_terms = {}
        quadratic_terms = {}

        # 1. Minimize the total weight of selected edges
        # Root edges: Only consider one direction (v0 -> u)
        for u in self.G.neighbors(self.root):
            edge_var = f'e_{self.root}_{u}'
            weight = self.G[self.root][u]['weight']
            linear_terms[edge_var] = linear_terms.get(edge_var, 0) + weight

        # Non-root edges: Include both directions (u -> v and v -> u)
        for u, v, w in self.G.edges(data='weight'):
            if u != self.root and v != self.root:
                edge_var_uv = f'e_{u}_{v}'
                edge_var_vu = f'e_{v}_{u}'
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
                    var_name_x_uv = f'x_{u}_{v}'
                    var_name_x_vw = f'x_{v}_{w}'
                    var_name_x_uw = f'x_{u}_{w}'

                    # Ensure all required variables exist
                    variables = [var_name_x_uv, var_name_x_vw, var_name_x_uw]
                    if not all(var in self.qubo.variables_index for var in variables):
                        continue

                    # Accumulate linear term: +P_I * x_{u,w}
                    linear_terms[var_name_x_uw] = linear_terms.get(var_name_x_uw, 0) + self.P_I

                    # Accumulate quadratic terms
                    quadratic_terms[tuple(sorted([var_name_x_uv, var_name_x_vw]))] = \
                        quadratic_terms.get(tuple(sorted([var_name_x_uv, var_name_x_vw])), 0) + self.P_I
                    quadratic_terms[tuple(sorted([var_name_x_uv, var_name_x_uw]))] = \
                        quadratic_terms.get(tuple(sorted([var_name_x_uv, var_name_x_uw])), 0) - self.P_I
                    quadratic_terms[tuple(sorted([var_name_x_uw, var_name_x_vw]))] = \
                        quadratic_terms.get(tuple(sorted([var_name_x_uw, var_name_x_vw])), 0) - self.P_I

        # Alignment Constraint (Term (ii))
        for (u, v) in self.G.edges():
            if u < v and u != self.root and v != self.root:
                var_name_e_uv = f'e_{u}_{v}'
                var_name_e_vu = f'e_{v}_{u}'
                var_name_x_uv = f'x_{u}_{v}'

                # Ensure all required variables exist
                variables = [var_name_e_uv, var_name_e_vu, var_name_x_uv]
                if not all(var in self.qubo.variables_index for var in variables):
                    continue

                # Accumulate linear term: +P_I * e_{u,v}
                linear_terms[var_name_e_uv] = linear_terms.get(var_name_e_uv, 0) + self.P_I

                # Accumulate quadratic terms
                quadratic_terms[tuple(sorted([var_name_e_uv, var_name_x_uv]))] = \
                    quadratic_terms.get(tuple(sorted([var_name_e_uv, var_name_x_uv])), 0) - self.P_I
                quadratic_terms[tuple(sorted([var_name_e_vu, var_name_x_uv]))] = \
                    quadratic_terms.get(tuple(sorted([var_name_e_vu, var_name_x_uv])), 0) + self.P_I

        # 3. Set the final objective function in the QUBO
        self.qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

    def add_constraints(self):
        """Add constraints to enforce the DCMST problem conditions."""
        # 1.1 and 1.2 are redundant restrictions, trying to improve the results.
        # ------------------------------------------------------------------------------------
        if self.redundancy:
            # 1.1 Edge count constraint: sum of selected edges = n - 1
            edge_count_constraint = {
                f'e_{u}_{v}': 1 for u, v in self.G.edges() if v != self.root
            }
            self.qubo.linear_constraint(
                linear=edge_count_constraint,
                sense='==',
                rhs=self.n - 1,
                name='edge_count_constraint'
            )

            # 1.2 Adicionar restrições para garantir e_{u,v} + e_{v,u} <= 1, quando não envolvem a raíz
            for u, v in self.G.edges:
                if u != self.root and v != self.root:
                    # Evitar redundância (adicionar apenas uma vez por par de arestas)
                    self.qubo.linear_constraint(
                        linear={f'e_{u}_{v}': 1, f'e_{v}_{u}': 1},
                        sense='<=',
                        rhs=1,  # Somente uma dessas variáveis pode ser 1 ao mesmo tempo
                        name=f'anti_symmetric_constraint_{u}_{v}'
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
            # max_degree = self.degree_constraints.get(v, self.n - 1)

            # Number of standard binary bits

            # -----------------------------
            #  1) Construct z_v as:
            #     z_v = sum_{i=0}^{k-1} 2^i z_{v,i} + (Delta+1 - 2^k)*z_{v,k}
            # -----------------------------
            all_terms = {}

            # (A) Standard bits (0..k-1)
            for i in range(self.binary_bits-1):
                bit_name = f"z_{v}_{i}"
                # Coefficient +2^i
                all_terms[bit_name] = all_terms.get(bit_name, 0.0) + (2**i)

            # (B) Special last bit (kth bit):
            #     (Delta+1 - 2^k) * z_{v,k}
            special_bit_name = f"z_{v}_{self.binary_bits-1}"
            all_terms[special_bit_name] = all_terms.get(special_bit_name, 0.0) + (
                (self.max_degree + 1) - 2**(self.binary_bits-1)
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

    def relax_problem(self, problem) -> QuadraticProgram:
        """Change all variables to continuous."""
        relaxed_problem = copy.deepcopy(problem)
        for variable in relaxed_problem.variables:
            if 'x' not in variable.name: 
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
    
    def validate_topological_states(self, n_variables):
        """
        Valida os estados possíveis para variáveis x_u,v para garantir que são acíclicos.

        Parâmetros:
        n_variables (int): Número de variáveis x_u,v no grafo (binomial(N-1, 2)).

        Retorna:
        List[str]: Lista de estados válidos no formato binário (e.g., ["000", "001"]).
        """
        valid_states = []
        # Gera todas as combinações possíveis de bits para n_variables
        all_states = product([0, 1], repeat=n_variables)
    
        for state in all_states:
            # Verifica se o estado é transitivo (condição para ser válido)
            if self.is_transitive(state, n_variables):
                # Adiciona o estado válido na lista como string binária
                valid_states.append("".join(map(str, state)))
        
        return valid_states


    def is_transitive(self, state, n_variables):
        """
        Verifica se um estado é transitivo, ou seja, se não cria ciclos.

        Parâmetros:
        state (tuple): Um estado das variáveis x_u,v (e.g., (0, 1, 0)).
        n_variables (int): Número de variáveis no estado.

        Retorna:
        bool: True se o estado é válido, False caso contrário.
        """
        # Mapeia variáveis x_u,v para uma matriz de ordenação
        matrix_size = int((1 + (1 + 8 * n_variables)**0.5) // 2)  # Número de vértices no grafo
        ordering_matrix = [[0] * matrix_size for _ in range(matrix_size)]
        
        # Preenche a matriz de ordenação com base no estado
        index = 0
        for u in range(matrix_size):
            for v in range(u + 1, matrix_size):
                ordering_matrix[u][v] = state[index]
                ordering_matrix[v][u] = 1 - state[index]
                index += 1

        # Verifica se a matriz representa uma ordenação acíclica (transitiva)
        for u in range(matrix_size):
            for v in range(matrix_size):
                for w in range(matrix_size):
                    if ordering_matrix[u][v] == 1 and ordering_matrix[v][w] == 1:
                        if ordering_matrix[u][w] != 1:
                            return False
        return True   

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
            format(i, f'0{n_bits}b')  # Formata o número como uma string binária de n_bits
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
        self.valid_e_uv_states = self.generate_states_excluding_specific(2, '11')
        self.valid_z_i_states = self.generate_z_states()

        for variable in self.qubo.variables:

            if 'e_' + str(self.root) in variable.name: 
                self.group_e_v0.append(idx)
            elif 'e' in variable.name:
                if idx not in group_e_uv:
                    group_e_uv.append(idx)
                    group_e_uv.append(idx+1)
                    init_qc = self.prepare_superposition(init_qc, self.valid_e_uv_states, group_e_uv)
                    self.groups_e_uv.append(group_e_uv)
                else:
                    group_e_uv = []

            elif 'z' in variable.name:
                if idx not in group_z_i:
                    for i in range(self.binary_bits):
                        group_z_i.append(idx+i)
                    self.groups_z_i.append(group_z_i)
                    init_qc = self.prepare_superposition(init_qc, self.valid_z_i_states, group_z_i)
                else:
                    group_z_i = []
            elif 'x' in variable.name:
                self.group_x.append(idx)

            idx += 1

        self._num_x = len(self.group_x)
        self._num_e_v0 = len(self.group_e_v0)
        # I will use this valid_x_states list, ['000000', '000001', ...], to prepare the initial state
        self.valid_x_states = self.validate_topological_states(self._num_x)
        init_qc = self.prepare_superposition(init_qc, self.valid_x_states, self.group_x)

        # I'm not certain if this worth the cost
        # valid_e_v0_states = self.generate_states_excluding_specific(self._num_e_v0, '0'*self._num_e_v0)
        # self.prepare_superposition(init_qc, valid_e_v0_states, group_e_v0)
        for id in self.group_e_v0:
            init_qc.h(id)

        # init_qc.draw(output="mpl", style="clifford") 
        init_qc.draw(output="mpl").savefig("init_qc.png")

        # print(init_qc)

        # print('Até aqui eu fui. Encerrei a preparação do estado inicial.')

        return init_qc


    # def mixer_LogicalX(self, hamiltonian, qubit_indices, beta, circuit):
    #     """
    #     Implementa um mixer no Qiskit baseado no Hamiltoniano fornecido com trotterização otimizada.

    #     Args:
    #         hamiltonian (list of tuples): Lista de termos do Hamiltoniano.
    #                                     Cada termo é (pauli_string, restrições),
    #                                     onde restrições é uma lista de (pauli_string, coeficiente).
    #         qubit_indices (list of int): Índices dos qubits sobre os quais o mixer deve atuar.
    #         beta (float): Parâmetro do QAOA associado ao mixer.
    #         circuit (QuantumCircuit): Circuito utilizado para preparar o mixer.

    #     Returns:
    #         QuantumCircuit: Circuito implementando o mixer.
    #     """
    #     # Verificar consistência dos índices de qubits
    #     num_qubits = len(qubit_indices)

    #     # Para cada termo no Hamiltoniano
    #     for main_pauli_string, constraints in hamiltonian:
    #         for constraint_pauli_string, coef in constraints:
    #             # Combinar o termo principal com a restrição
    #             combined_pauli_string = ''.join(
    #                 main_pauli_string[i] if main_pauli_string[i] != 'I' else constraint_pauli_string[i]
    #                 for i in range(len(main_pauli_string))
    #             )

    #             # Mapear operadores para os qubits
    #             assert len(combined_pauli_string) == num_qubits, "O tamanho da string de Pauli deve corresponder ao número de qubits."

    #             # Verificar se o termo pode ser implementado diretamente com RX
    #             if all(op in {'X', 'I'} for op in combined_pauli_string):
    #                 for i, op in enumerate(combined_pauli_string):
    #                     if op == 'X':
    #                         circuit.rx(-2 * beta * coef, qubit_indices[i])  # Aplicar RX diretamente
    #             else:
    #                 # Termos mais complexos: usar CNOTs, Hadamards e RZ
    #                 involved_qubits = []
    #                 for i, op in enumerate(combined_pauli_string):
    #                     if op == 'X':
    #                         circuit.h(qubit_indices[i])
    #                         involved_qubits.append(qubit_indices[i])
    #                     elif op == 'Z':
    #                         involved_qubits.append(qubit_indices[i])
    #                     elif op == 'I':
    #                         continue

    #                 # Aplicar controle para os qubits envolvidos
    #                 if len(involved_qubits) > 1:
    #                     # Aplicar CNOTs para criar a cadeia de controles
    #                     for i in range(len(involved_qubits) - 1):
    #                         circuit.cx(involved_qubits[i], involved_qubits[i + 1])

    #                 # Aplicar a rotação parametrizada no último qubit da cadeia
    #                 rotation_qubit = involved_qubits[-1]
    #                 circuit.rz(-2 * beta * coef, rotation_qubit)

    #                 # Reverter os CNOTs
    #                 if len(involved_qubits) > 1:
    #                     for i in reversed(range(len(involved_qubits) - 1)):
    #                         circuit.cx(involved_qubits[i], involved_qubits[i + 1])

    #                 # Reverter Hadamard gates
    #                 for i, op in enumerate(combined_pauli_string):
    #                     if op == 'X':
    #                         circuit.h(qubit_indices[i])

    #     return circuit

    def mixer_LogicalX(self, hamiltonian, qubit_indices, beta, circuit):
        backend = self.configure_backend()

        num_qubits = len(qubit_indices)
        ancilla_cicuit = QuantumCircuit(num_qubits)

        sp_op_total = SparsePauliOp.from_list([('I' * num_qubits, 0.0)])

        for main_pauli_string, constraints in hamiltonian:
            for constraint_pauli_string, coef in constraints:
                combined_pauli_string = ''.join(
                    main_pauli_string[i] if main_pauli_string[i] != 'I'
                    else constraint_pauli_string[i]
                    for i in range(num_qubits)
                )
                sp_op_term = SparsePauliOp.from_list([(combined_pauli_string, coef)])
                sp_op_total = sp_op_total + sp_op_term

        # Use LieTrotter instead of SuzukiTrotter
        trotter = LieTrotter()
        evolution_gate = PauliEvolutionGate(sp_op_total, time=beta, synthesis=trotter)

        circuit.append(evolution_gate, qubit_indices)

        return circuit


    def mixer_customized(self):
        mixer_circuit = QuantumCircuit(len(self.qubo.variables))
        beta = Parameter("β")

        # For the e_v0_u variables
        for id in self.group_e_v0:
            mixer_circuit.rx(-2*beta, id)

        # For the e_uv and e_vu variables
        # Needs to search in the hamiltonian_mixers.cvs using the valid_states
        df = pd.read_csv('hamiltonian_mixers.csv')

        for ids in self.groups_e_uv:
            # Procurar o Hamiltoniano correspondente ao estado
            filtered_row = df[df['states'] == str(self.valid_e_uv_states)]
            if not filtered_row.empty:  
                hamiltonian = eval(filtered_row.iloc[0]['hamiltonian'])  
                print(hamiltonian, ids)
                mixer_circuit = self.mixer_LogicalX(hamiltonian, ids, beta, mixer_circuit)

            else:
                print(f"Estado {self.valid_e_uv_states} referente às variaveis e_uv não encontrado no arquivo CSV.")
                sys.exit()

        # For the x_uv
        # Needs to search in the hamiltonian_mixers.cvs using the valid_states
        filtered_row = df[df['states'] == str(self.valid_x_states)]
        
        if not filtered_row.empty:  # Verificar se encontrou o estado
            hamiltonian = eval(filtered_row.iloc[0]['hamiltonian'])  # Avaliar o Hamiltoniano como objeto Python
            print(hamiltonian, self.group_x)
            
            # Chamar a função mixer_LogicalX com os dados encontrados
            mixer_circuit = self.mixer_LogicalX(hamiltonian, self.group_x, beta, mixer_circuit)

        else:
            print(f"Estado {self.valid_x_states} referente às variaveis x_uv não encontrado no arquivo CSV.")
            sys.exit()        

        # For the z_vi
        # Needs to search in the hamiltonian_mixers.cvs using the valid_states
        for ids in self.groups_z_i:
            filtered_row = df[df['states'] == str(self.valid_z_i_states)]
            
            if not filtered_row.empty:  # Verificar se encontrou o estado
                hamiltonian = eval(filtered_row.iloc[0]['hamiltonian'])  # Avaliar o Hamiltoniano como objeto Python
                print(hamiltonian, ids)
                
                # Chamar a função mixer_LogicalX com os dados encontrados
                mixer_circuit = self.mixer_LogicalX(hamiltonian, ids, beta, mixer_circuit)
            else:
                print(f"Estado {self.valid_x_states} referente às variaveis z_vi não encontrado no arquivo CSV.")
                sys.exit()  

        return mixer_circuit        


    def solve_problem(self, optimizer, p=1, parameters = None):
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
        optimum_parameters = None
        def callback(eval_count, params, mean, std):
            global optimum_parameters
            print(f"Eval count: {eval_count}, Parameters: {params}, Mean + offset: {mean + offset}")
            optimum_parameters = params
        # Set up QAOA with the callback

        np.random.seed(self.seed)

        # Define the seed that will be used in the optimization process 
        algorithm_globals.random_seed = self.seed

        # Generate initial parameters using the seed
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * p) 
        if not self.VQE:       
            if (self.mixer is not None) and  (self.initial_state is not None):
                if self.mixer == 'Warm':  
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
                elif self.mixer == 'LogicalX':
                    self.initial_state = self.initial_state_OHE()
                    self.mixer = self.mixer_customized()


                optimized_mixer = transpile(self.mixer, backend, optimization_level=3)

                optimized_mixer.draw(output="mpl").savefig("mixer_circuit.png")

                # Create a custom pass manager
                # pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

                # Transpile the circuit
                # optimized_mixer = pm.run(self.mixer)             

                qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=p, initial_point=initial_params, 
                                mixer=optimized_mixer, initial_state=self.initial_state ,callback=callback)
            else:
                qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=p, initial_point=initial_params, callback=callback)
                if self.warm_start:
                    qaoa_mes = WarmStartQAOAOptimizer(
                        pre_solver=CplexOptimizer(), relax_for_pre_solver=True, qaoa=qaoa_mes, epsilon=0.0, penalty=self.P_I
                    )

            if not self.warm_start:
                qaoa = MinimumEigenOptimizer(qaoa_mes, penalty=self.P_I)
            else:
                qaoa = qaoa_mes
        else:
            # 1) Build the parameterized ansatz for VQE
            from qiskit.circuit.library import TwoLocal

            ansatz = TwoLocal(len(self.qubo.variables), rotation_blocks='ry', entanglement_blocks='cx', entanglement='linear', reps=p)
            initial_parameters = np.random.random(ansatz.num_parameters)

            sampling_vqe = SamplingVQE(
                sampler=sampler,
                ansatz=ansatz,
                optimizer=optimizer,
                callback=callback  # if you have a callback function defined
            )            

            start_time = time.time()
            result = sampling_vqe.compute_minimum_eigenvalue(operator=qubo_ops)
            end_time = time.time()

            self.execution_time = end_time - start_time
            print(f"VQE ground state energy (no offset): {result.eigenvalue.real:.5f}")
            print(f"VQE ground state energy (with offset): {(result.eigenvalue + offset).real:.5f}")
            print(f"Execution time: {self.execution_time:.4f} seconds")

            # 4) Retrieve the optimal parameters
            optimal_params = result.optimal_point
            print("Optimal parameters:", optimal_params)

            # Bind the optimal parameters to the ansatz
            bound_ansatz = ansatz.assign_parameters(optimal_params)

            # Sample the circuit using the sampler
            sampled_result = sampler.run(bound_ansatz, shots=1024).result()
            quasi_dist = sampled_result.quasi_dists[0]  # Assuming a single circuit

            shots = 1024
            float_counts = quasi_dist.binary_probabilities()
            counts = {state: int(round(prob * shots)) for state, prob in float_counts.items()}


            # 7) Return or store them
            print(counts)
            return counts

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
        self.num_qubits = len(self.qubo.variables)
        print(f"Number of qubits required: {self.num_qubits}")

