"""
RQ2 Graph Generator: Sliding-N Strategy for FWHT vs DFS Comparison.

Produces three graph families with controlled genus (g) and node count (N):
  1. Sparse  (Havel-Hakimi style) : N=15-30, g=5-14
  2. Medium  (Tangled Web)        : N=8-10,  g=6-12
  3. Dense   (Extreme overlap)    : N=6-7,   g=6-10

FWHT runtime depends primarily on g (and m), so FWHT curves across families
nearly overlap.  DFS runtime depends heavily on the number of Eulerian
circuits, which grows with edge density — so DFS curves diverge sharply.

Output: DIMACS .gr format
"""

import os
import random
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

random.seed(42)  # reproducibility


def ensure_even_degrees(edges, n):
    """Fix parity: pair up odd-degree nodes and add an edge between each pair."""
    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    odd_nodes = [v for v in range(1, n + 1) if deg[v] % 2 != 0]
    random.shuffle(odd_nodes)
    added = []
    while len(odd_nodes) >= 2:
        u = odd_nodes.pop()
        v = odd_nodes.pop()
        added.append((u, v))
    return edges + added


# ---------- Family 1: Sparse (large N, low density) ----------
def generate_sparse_graph(target_n, target_g, output_filename):
    """
    Sparse Eulerian graph: start with a Hamiltonian cycle, then add
    long random cycles (length 5-8) using existing nodes to reach
    the target genus.
    """
    edges = []

    # Hamiltonian cycle (g = 1)
    for i in range(1, target_n):
        edges.append((i, i + 1))
    edges.append((target_n, 1))
    current_g = 1

    # Add long cycles to slowly increase genus
    while current_g < target_g:
        max_L = min(8, target_n, target_g - current_g + 1)
        min_L = min(5, max_L)
        if min_L < 3:
            break
        L = random.randint(min_L, max_L)
        if current_g + L - 1 > target_g:
            L = target_g - current_g + 1
            if L < 3:
                break
        nodes = random.sample(range(1, target_n + 1), L)
        for i in range(L):
            edges.append((nodes[i], nodes[(i + 1) % L]))
        current_g += (L - 1)

    edges = ensure_even_degrees(edges, target_n)
    final_m = len(edges)
    actual_g = final_m - target_n + 1

    with open(output_filename, 'w') as f:
        f.write(f"c Sparse Eulerian graph N={target_n} g={actual_g}\n")
        f.write(f"p edge {target_n} {final_m}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    print(f"  [Sparse]  {os.path.basename(output_filename)}: "
          f"N={target_n}, M={final_m}, g={actual_g}")
    return actual_g


# ---------- Family 2: Medium density (Tangled Web) ----------
def generate_tangled_web(target_n, target_g, output_filename):
    """
    Medium-density Eulerian graph: overlapping short cycles (length 3-4)
    on a moderate node set.
    """
    edges = []

    # Base cycle
    for i in range(1, target_n):
        edges.append((i, i + 1))
    edges.append((target_n, 1))
    current_g = 1

    # Overlay short random cycles
    while current_g < target_g:
        L = random.choice([3, 4])
        if current_g + L - 1 > target_g:
            L = target_g - current_g + 1
            if L < 3:
                break
        nodes = random.sample(range(1, target_n + 1), L)
        for i in range(L):
            edges.append((nodes[i], nodes[(i + 1) % L]))
        current_g += (L - 1)

    edges = ensure_even_degrees(edges, target_n)
    final_m = len(edges)
    actual_g = final_m - target_n + 1

    with open(output_filename, 'w') as f:
        f.write(f"c Tangled Web Eulerian graph N={target_n} g={actual_g}\n")
        f.write(f"p edge {target_n} {final_m}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    print(f"  [Medium]  {os.path.basename(output_filename)}: "
          f"N={target_n}, M={final_m}, g={actual_g}")
    return actual_g


# ---------- Family 3: Dense (minimal N, maximum overlap) ----------
def generate_dense_graph(target_n, target_g, output_filename):
    """
    Extremely dense Eulerian graph: triangles on a tiny node set
    to maximize DFS branching factor.
    """
    edges = []

    # Base cycle
    for i in range(1, target_n):
        edges.append((i, i + 1))
    edges.append((target_n, 1))
    current_g = 1

    # Flood with triangles (each adds g += 2)
    while current_g < target_g:
        L = 3
        if current_g + L - 1 > target_g:
            break
        nodes = random.sample(range(1, target_n + 1), L)
        for i in range(L):
            edges.append((nodes[i], nodes[(i + 1) % L]))
        current_g += (L - 1)

    edges = ensure_even_degrees(edges, target_n)
    final_m = len(edges)
    actual_g = final_m - target_n + 1

    with open(output_filename, 'w') as f:
        f.write(f"c Dense Eulerian graph N={target_n} g={actual_g}\n")
        f.write(f"p edge {target_n} {final_m}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    print(f"  [Dense]   {os.path.basename(output_filename)}: "
          f"N={target_n}, M={final_m}, g={actual_g}")
    return actual_g


# ======================== Main ========================
if __name__ == "__main__":
    output_dir = os.path.join(REPO_ROOT, "data", "rq2_graphs")
    os.makedirs(output_dir, exist_ok=True)

    # Family 1: Sparse — N slides from 15 to 30 as g grows
    sparse_config = [
        (5, 15), (6, 16), (7, 17), (8, 18), (9, 19),
        (10, 20), (11, 22), (12, 24), (13, 27), (14, 30),
    ]

    # Family 2: Medium — N slides from 8 to 10
    medium_config = [
        (6, 8), (7, 8), (8, 9), (9, 9), (10, 10), (11, 10), (12, 10),
    ]

    # Family 3: Dense — N fixed at 6-7
    dense_config = [
        (6, 7), (7, 7), (8, 7), (9, 6), (10, 6),
    ]

    print("=" * 60)
    print("RQ2 Graph Generator: Sliding-N Strategy")
    print("=" * 60)

    print("\n--- Family 1: Sparse ---")
    for g, n in sparse_config:
        fname = os.path.join(output_dir, f"sparse_n{n}_g{g}.gr")
        generate_sparse_graph(target_n=n, target_g=g, output_filename=fname)

    print("\n--- Family 2: Medium (Tangled Web) ---")
    for g, n in medium_config:
        fname = os.path.join(output_dir, f"medium_n{n}_g{g}.gr")
        generate_tangled_web(target_n=n, target_g=g, output_filename=fname)

    print("\n--- Family 3: Dense ---")
    for g, n in dense_config:
        fname = os.path.join(output_dir, f"dense_n{n}_g{g}.gr")
        generate_dense_graph(target_n=n, target_g=g, output_filename=fname)

    total = len(sparse_config) + len(medium_config) + len(dense_config)
    print(f"\n>>> Generated {total} graphs in: {output_dir}")
