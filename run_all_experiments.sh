#!/bin/bash
# =============================================================================
# FWHT Paper - Complete Experiment Runner
# Runs both FWHT solver and DFS baseline on all selected graph datasets
# =============================================================================
#
# Usage:
#   bash run_all_experiments.sh
#
# Prerequisites:
#   - GCC with C++17 and OpenMP support
#   - Graph files in datasets/selected_gr/
#
# Output:
#   results/fwht_MUTAG.csv, results/fwht_AIDS.csv, etc.
#   results/dfs_MUTAG.csv, results/dfs_AIDS.csv, etc.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 1: Compile solvers
# =============================================================================
echo "=========================================="
echo "Step 1: Compiling solvers..."
echo "=========================================="

echo "  Compiling FWHT solver..."
g++ -O3 -std=c++17 -funroll-loops -fopenmp \
    fwht_solver.cpp -o fwht_solver -lstdc++fs 2>/dev/null \
  || g++ -O3 -std=c++17 -funroll-loops -fopenmp \
    fwht_solver.cpp -o fwht_solver
echo "  -> fwht_solver OK"

echo "  Compiling DFS solver..."
g++ -O3 -std=c++17 -funroll-loops -fopenmp \
    dfs_solver.cpp -o dfs_solver -lstdc++fs 2>/dev/null \
  || g++ -O3 -std=c++17 -funroll-loops -fopenmp \
    dfs_solver.cpp -o dfs_solver
echo "  -> dfs_solver OK"

echo ""

# =============================================================================
# Step 2: RQ5 - FWHT on all real-world datasets
# =============================================================================
echo "=========================================="
echo "Step 2: RQ5 - FWHT on real-world graphs"
echo "=========================================="

GR_BASE="datasets"

for dataset in MUTAG AIDS PTC_MR infrastructure; do
    ds_dir="$GR_BASE/$dataset"
    if [ -d "$ds_dir" ]; then
        out_csv="$RESULTS_DIR/fwht_${dataset}.csv"
        echo "  Running FWHT on $dataset -> $out_csv"
        ./fwht_solver "$ds_dir" "$out_csv"
        echo ""
    else
        echo "  [SKIP] $ds_dir not found"
    fi
done

# =============================================================================
# Step 3: RQ2-enhanced - DFS baseline on all datasets
# =============================================================================
echo "=========================================="
echo "Step 3: RQ2 - DFS baseline comparison"
echo "=========================================="

for dataset in MUTAG AIDS PTC_MR infrastructure; do
    ds_dir="$GR_BASE/$dataset"
    if [ -d "$ds_dir" ]; then
        out_csv="$RESULTS_DIR/dfs_${dataset}.csv"
        echo "  Running DFS on $dataset -> $out_csv"
        ./dfs_solver "$ds_dir" "$out_csv"
        echo ""
    else
        echo "  [SKIP] $ds_dir not found"
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "All experiments complete!"
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.csv 2>/dev/null || echo "(no CSV files found)"
echo "=========================================="
