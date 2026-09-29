# Quantum-DCMST

Quantum implementations for the **Degree-Constrained Minimum Spanning Tree (DCMST)** problem using Qiskit, including QAOA/VQE-based experiments and alternative initialization/mixer strategies.

## Overview

The code implements a quantum-optimization workflow for DCMST. The DCMST-specific QUBO/Ising construction is based on the mathematical formulation introduced by Alex Fowler for the degree-constrained minimum spanning tree problem.

The main implementation is contained in `qubo_problem.py`, which defines:

- binary variables representing directed edge selection;
- ordering variables used to enforce acyclicity through a topological-order construction;
- binary degree-counter variables;
- the weighted objective function;
- DCMST feasibility constraints and their QUBO conversion;
- QAOA and VQE execution utilities;
- warm-start initialization and mixer construction;
- customized state-preparation/mixer routines used in the experiments.

## Mathematical formulation

The problem-specific formulation implemented in this repository follows:

> A. Fowler, **“Improved QUBO Formulations for D-Wave Quantum Computing,”**  
> Master’s thesis, Department of Computer Science, University of Auckland, 2017.

In particular, Section 5.1.2 of Fowler’s thesis introduces an improved QUBO formulation for DCMST using:

- directed edge variables $e_{u,v}$;
- ordering variables $x_{u,v}$;
- degree-counter variables $z_{v,i}$;
- a penalty coefficient
  $$
  P_I = (|V|-1)m + 1,
  $$
  where $m$ is the maximum edge cost.

The implementation in `qubo_problem.py` translates this mathematical formulation into Qiskit’s `QuadraticProgram` representation and subsequently converts it to QUBO/Ising form for variational quantum optimization.

## Repository structure

- `qubo_problem.py` — DCMST formulation, QUBO/Ising conversion, QAOA/VQE execution, mixers, initialization and helper routines.
- `opt.py` — experiment/optimization entry point.
- `graph.py` — graph-related utilities.
- `config.py` — configuration handling.
- `main.py` and `main_X_Standard_Experiment.py` — experiment scripts.
- `qopt/` — optimization-related helper code.
- `experiments/` — experiment configurations/results structure.
- `experiments_example/` — example experiment configurations.
- `requirements.txt` — Python dependencies.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Some execution modes require additional external software or credentials, for example IBM Quantum access and/or IBM CPLEX.

## Usage

A typical optimization run can be started with:

```bash
python opt.py -c path_to_config.yaml
```

The optimization results are written to the experiment results directory configured by the project.

## Code provenance and third-party material

### DCMST formulation

The DCMST-specific mathematical model in `qubo_problem.py` is an implementation of Fowler’s QUBO formulation cited above. Fowler is the source of the mathematical formulation; this repository provides a Qiskit implementation and the quantum-optimization workflow used in the accompanying IJCNN 2025 study.

### Qiskit / IBM Quantum code

The repository depends extensively on the Qiskit ecosystem.

In addition, portions of the QAOA execution workflow in `qubo_problem.py` were adapted from IBM Quantum / Qiskit documentation examples. In particular, the structure of the estimator-based cost function, layout application, Runtime Estimator/Sampler usage, transpilation workflow, and the dynamical-decoupling/twirling configuration follow the IBM Quantum **“Quantum approximate optimization algorithm”** tutorial.

Relevant upstream source:

- IBM Quantum / Qiskit Documentation, *Quantum approximate optimization algorithm*:  
  https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm

The Qiskit documentation repository states that **code snippets in documentation examples are licensed under the Apache License 2.0**:

- https://github.com/Qiskit/documentation

### Warm-start QAOA

The warm-start initialization and mixer construction use the standard warm-start QAOA prescription implemented and documented in Qiskit Optimization, including the relaxed solution \(c_i^*\), the rotation

$$
\theta_i = 2\arcsin\sqrt{c_i^*},
$$

and the corresponding warm-start mixer.

Relevant references:

- Qiskit Optimization, *Warm-starting quantum optimization*:  
  https://qiskit-community.github.io/qiskit-optimization/tutorials/10_warm_start_qaoa.html

- D. J. Egger, J. Mareček, and S. Woerner,  
  **“Warm-starting quantum optimization,”** *Quantum*, vol. 5, p. 479, 2021.  
  https://doi.org/10.22331/q-2021-06-17-479

Qiskit Optimization is distributed under the Apache License 2.0.

## Citation

If you use this repository in academic work, please cite:

```bibtex
@inproceedings{Carmo2025QuantumDCMST,
  author    = {Rafael Sim{\~o}es do Carmo and
               Marcos Cleison Silva Santana and
               Felipe Fernandes Fanchini and
               Kelton A. P. Costa and
               Weslley Santana Rosalem and
               Jo{\~a}o Paulo Papa},
  title     = {Quantum Approaches for Degree-Constrained Minimum Spanning Tree Computation},
  booktitle = {2025 International Joint Conference on Neural Networks (IJCNN)},
  pages     = {1--8},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/IJCNN64981.2025.11228458}
}
```

For the underlying DCMST QUBO formulation, please also cite:

```bibtex
@mastersthesis{Fowler2017ImprovedQUBO,
  author = {Alex Fowler},
  title  = {Improved QUBO Formulations for D-Wave Quantum Computing},
  school = {University of Auckland},
  year   = {2017},
  type   = {Master's thesis}
}
```

## License

This repository is distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

The Apache-2.0 choice is compatible with the identified Qiskit code material incorporated or adapted in this repository, which is itself distributed under Apache-2.0. Third-party components remain subject to their respective licenses and attribution requirements.
