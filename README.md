# Counting Eulerian Circuits via FWHT with FPT Parameterization by Genus

This repository contains the source code and experimental scripts for reproducing all results in the paper. The algorithm counts Eulerian circuits in undirected graphs using the **Fast Walsh-Hadamard Transform (FWHT)** over twisted adjacency matrices, achieving **FPT complexity O(2^g · m³ · log m)** parameterized by the circuit rank (genus) *g*.

## Repository Structure

```
eulerian-fwht/
├── src/
│   ├── fwht_solver.cpp      # FWHT-based parallel Eulerian circuit counter (OpenMP)
│   └── dfs_solver.cpp        # DFS backtracking baseline (for RQ2 comparison)
├── generators/
│   ├── gen_rq1.py            # Graph generator for RQ1 (time vs genus)
│   ├── gen_rq2.py            # Graph generator for RQ2 (FWHT vs DFS)
│   ├── gen_rq3.py            # Graph generator for RQ3 (time vs graph size)
│   └── gen_rq4.py            # Graph generator for RQ4 (parallel scaling)
├── plotting/
│   ├── plot_rq1.py           # RQ1 figures (time & circuit count vs g)
│   ├── plot_rq1_supp.py      # RQ1 supplementary (circuits vs edges at fixed g)
│   ├── plot_rq2.py           # RQ2 figures (FWHT vs DFS comparison)
│   ├── plot_rq3.py           # RQ3 figures (time vs M at fixed g)
│   └── plot_rq4.py           # RQ4 figures (speedup, efficiency, Amdahl's law)
├── scripts/
│   ├── run_rq3.ps1           # End-to-end RQ3 experiment script (PowerShell)
│   └── run_rq4.ps1           # End-to-end RQ4 experiment script (PowerShell)
├── .gitignore
└── README.md
```

The `data/`, `results/`, and `figures/` directories are created at runtime.

## Prerequisites

### C++ Compiler
- **g++ 11+** with C++17 filesystem support
- **OpenMP** for multi-threading
- **AVX2** instruction set support (most x86-64 CPUs from 2013+)

### Python 3.8+
```bash
pip install numpy matplotlib scipy pandas networkx
```

## Compilation

```bash
# FWHT solver (main algorithm)
g++ -O3 -funroll-loops -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt \
    -fopenmp src/fwht_solver.cpp -o fwht_solver

# DFS solver (baseline for RQ2)
g++ -O3 -funroll-loops -fopenmp src/dfs_solver.cpp -o dfs_solver
```

### Optional: GMP for arbitrary precision
```bash
g++ -O3 -funroll-loops -mavx2 -fopenmp -DUSE_GMP \
    src/fwht_solver.cpp -o fwht_solver -lgmp -lgmpxx
```

## Usage

```bash
# Run solver on a directory of .gr files
./fwht_solver <graph_directory> [output_csv]

# Example
./fwht_solver data/rq1_fig1_graphs results/rq1_results.csv
```

## Reproducing Experiments

### RQ1: Exponential Scaling with Circuit Rank

```bash
# 1. Generate graphs
python generators/gen_rq1.py

# 2. Run solver
./fwht_solver data/rq1_fig1_graphs results/eulerian_RQ1_results.csv

# 3. Generate figures (uses embedded data from paper runs)
python plotting/plot_rq1.py
python plotting/plot_rq1_supp.py
```

### RQ2: FWHT vs DFS Comparison

```bash
# 1. Generate graphs (3 density families)
python generators/gen_rq2.py

# 2. Run both solvers
./fwht_solver data/rq2_graphs results/eulerian_RQ2_FWHT_results.csv
./dfs_solver  data/rq2_graphs results/eulerian_RQ2_DFS_results.csv

# 3. Generate comparison figures
python plotting/plot_rq2.py
```

### RQ3: Time vs Graph Size at Fixed Genus

```bash
# Automated (PowerShell):
powershell -ExecutionPolicy Bypass -File scripts/run_rq3.ps1

# Or manually:
python generators/gen_rq3.py
./fwht_solver data/rq3_graphs results/eulerian_RQ3_results.csv
python plotting/plot_rq3.py
```

### RQ4: Parallel Scaling (OpenMP)

```bash
# Automated (PowerShell):
powershell -ExecutionPolicy Bypass -File scripts/run_rq4.ps1

# Or manually:
python generators/gen_rq4.py
# Run with varying OMP_NUM_THREADS (1, 2, 4, 8, 16, 32)
OMP_NUM_THREADS=1 ./fwht_solver data/rq4_graphs results/rq4_t1.csv
# ... repeat for each thread count
python plotting/plot_rq4.py
```

## Graph Format (DIMACS)

Input graphs use the DIMACS `.gr` format:

```
c Comment line (optional)
p edge <num_vertices> <num_edges>
e <u> <v>
e <u> <v>
...
```

Vertices are **1-indexed**. All graphs must be connected and Eulerian (every vertex has even degree).

## Timeout Behaviour

Both solvers support a 600-second timeout per graph:
- **FWHT solver**: Records partial results with completion ratio (e.g., `PARTIAL(512/4096)`)
- **DFS solver**: Records partial raw trail count for extrapolation

## License

This code is provided for academic review and reproducibility purposes.
