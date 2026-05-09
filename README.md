# FWHT-Based Exact Eulerian Circuit Counting

An efficient implementation of the Luo homological trace formula for exact Eulerian circuit counting in undirected graphs, using the Fast Walsh-Hadamard Transform (FWHT) for spectral decoding.

## Overview

Counting Eulerian circuits in undirected graphs is #P-complete. This implementation exploits the algebraic structure of the circuit rank (first Betti number) to achieve FPT complexity O(2^g * m^3 log m), where g is the circuit rank and m is the number of edges.

## Repository Structure

```
FWHT0329.tex          # Paper source (ACM JEA format)
references.bib        # BibTeX bibliography
figures (*.png)       # Paper figures (RQ1-RQ4)
fwht_solver.cpp       # FWHT Euler circuit solver (main algorithm)
dfs_solver.cpp        # DFS baseline with Fleury bridge pruning
run_all_experiments.sh # One-click experiment runner
datasets/             # 61 real-world graph files (DIMACS .gr format)
  MUTAG/              # 15 molecular graphs (circuit rank 4-11)
  AIDS/               # 15 molecular graphs (circuit rank 3-10)
  PTC_MR/             # 15 molecular graphs (circuit rank 3-10)
  infrastructure/     # 16 power grid + road network subgraphs (cr 7-18)
results/              # Experiment CSV results (RQ1-RQ5)
scripts/              # Plotting and graph generation scripts
logs/                 # Experiment execution logs
```

## Building

```bash
# FWHT solver (requires OpenMP)
g++ -O3 -std=c++17 -funroll-loops -fopenmp fwht_solver.cpp -o fwht_solver

# DFS baseline (Fleury bridge pruning)
g++ -O3 -std=c++17 -funroll-loops dfs_solver.cpp -o dfs_solver
```

## Usage

```bash
# Run on a directory of .gr files
./fwht_solver datasets/MUTAG results/fwht_MUTAG.csv
./dfs_solver  datasets/MUTAG results/dfs_MUTAG.csv

# Run all experiments
bash run_all_experiments.sh
```

## Key Results

- FWHT outperforms DFS backtracking for circuit rank g >= 12 (density-dependent crossover)
- Speedup reaches 177x on infrastructure graphs at g = 14
- All 61 real-world instances solved, including g = 18 in under 50 seconds
- Near-linear OpenMP parallel speedup for moderate problem sizes

## Citation

If you use this code, please cite the paper (to appear).

## License

MIT
