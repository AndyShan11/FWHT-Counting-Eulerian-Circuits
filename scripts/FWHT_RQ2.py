import os
import random

# ============================================================================
# RQ2 Graph Generator: Sliding-N Strategy + Three Graph Families
# ============================================================================
# Goal: For each target genus g, choose the smallest N that lets DFS finish
#        in reasonable time, producing three families of curves:
#   1. Sparse  (Havel-Hakimi style):  N=15-30, g=5-14
#   2. Medium  (Tangled Web):         N=8-10,  g=6-12
#   3. Dense   (Extreme overlap):     N=6-7,   g=6-10
# FWHT curves nearly overlap (depends on g only); DFS curves diverge sharply.
# ============================================================================

random.seed(42)  # Reproducibility


def ensure_even_degrees(edges, n):
    """Fix parity: for every odd-degree node, add a self-loop-free edge."""
    from collections import Counter
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


# ---------- Family 1: Sparse Graph (Havel-Hakimi style) ----------
def generate_sparse_graph(target_n, target_g, output_filename):
    """
    Sparse Eulerian graph: large N, low density.
    Start with a Hamiltonian cycle, then add long random cycles
    (length 5-8) using existing nodes to reach target genus.
    """
    edges = []

    # 1. Hamiltonian cycle -> connectivity + g=1
    for i in range(1, target_n):
        edges.append((i, i + 1))
    edges.append((target_n, 1))
    current_g = 1

    # 2. Add long cycles to slowly increase genus
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

    # 3. Ensure all degrees are even (Eulerian requirement)
    edges = ensure_even_degrees(edges, target_n)

    final_m = len(edges)
    actual_g = final_m - target_n + 1

    with open(output_filename, 'w') as f:
        f.write(f"c Sparse Eulerian Graph (Havel-Hakimi style)\n")
        f.write(f"c N={target_n}, M={final_m}, Genus={actual_g}\n")
        f.write(f"p edge {target_n} {final_m}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    print(f"  [Sparse]  {os.path.basename(output_filename)}: N={target_n}, M={final_m}, g={actual_g}")
    return actual_g


# ---------- Family 2: Medium Density (Tangled Web) ----------
def generate_tangled_web(target_n, target_g, output_filename):
    """
    Medium-density Eulerian graph: moderate N, overlapping short cycles.
    Uses triangles and squares on a small node set.
    """
    edges = []

    # 1. Base cycle
    for i in range(1, target_n):
        edges.append((i, i + 1))
    edges.append((target_n, 1))
    current_g = 1

    # 2. Overlay short random cycles (length 3-4)
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
        f.write(f"c Tangled Web Eulerian Graph (Medium Density)\n")
        f.write(f"c N={target_n}, M={final_m}, Genus={actual_g}\n")
        f.write(f"p edge {target_n} {final_m}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    print(f"  [Medium]  {os.path.basename(output_filename)}: N={target_n}, M={final_m}, g={actual_g}")
    return actual_g


# ---------- Family 3: Extreme Density ----------
def generate_dense_graph(target_n, target_g, output_filename):
    """
    Extremely dense Eulerian graph: minimal N, maximum edge overlap.
    Uses triangles exclusively on a tiny node set -> maximum DFS branching.
    """
    edges = []

    # 1. Base cycle
    for i in range(1, target_n):
        edges.append((i, i + 1))
    edges.append((target_n, 1))
    current_g = 1

    # 2. Flood with triangles (L=3, each adds g+=2)
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
        f.write(f"c Dense Eulerian Graph (Extreme Overlap)\n")
        f.write(f"c N={target_n}, M={final_m}, Genus={actual_g}\n")
        f.write(f"p edge {target_n} {final_m}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    print(f"  [Dense]   {os.path.basename(output_filename)}: N={target_n}, M={final_m}, g={actual_g}")
    return actual_g


# ============================================================================
# Main: Sliding-N configuration for each family
# ============================================================================
if __name__ == "__main__":
    output_dir = r"D:\cpp\.vscode\FWHTRQ2GR_KILLER"
    os.makedirs(output_dir, exist_ok=True)

    # --- Family 1: Sparse (large N, low density) ---
    # N slides from 15 to 30 as g grows; DFS explodes early due to high m
    sparse_config = [
        # (g, N)
        (5,  15),
        (6,  16),
        (7,  17),
        (8,  18),
        (9,  19),
        (10, 20),
        (11, 22),
        (12, 24),
        (13, 27),
        (14, 30),
    ]

    # --- Family 2: Medium density (Tangled Web) ---
    # N slides from 8 to 10
    medium_config = [
        (6,  8),
        (7,  8),
        (8,  9),
        (9,  9),
        (10, 10),
        (11, 10),
        (12, 10),
    ]

    # --- Family 3: Extreme density ---
    # N fixed at 6-7, only small g range feasible for DFS
    dense_config = [
        (6,  7),
        (7,  7),
        (8,  7),
        (9,  6),
        (10, 6),
    ]

    print("=" * 60)
    print("RQ2 Graph Generator: Sliding-N Strategy")
    print("=" * 60)

    print("\n--- Family 1: Sparse (Havel-Hakimi) ---")
    for g, n in sparse_config:
        fname = os.path.join(output_dir, f"sparse_n{n}_g{g}.gr")
        generate_sparse_graph(target_n=n, target_g=g, output_filename=fname)

    print("\n--- Family 2: Medium (Tangled Web) ---")
    for g, n in medium_config:
        fname = os.path.join(output_dir, f"medium_n{n}_g{g}.gr")
        generate_tangled_web(target_n=n, target_g=g, output_filename=fname)

    print("\n--- Family 3: Dense (Extreme Overlap) ---")
    for g, n in dense_config:
        fname = os.path.join(output_dir, f"dense_n{n}_g{g}.gr")
        generate_dense_graph(target_n=n, target_g=g, output_filename=fname)

    print("\n" + "=" * 60)
    total = len(sparse_config) + len(medium_config) + len(dense_config)
    print(f">>> Generated {total} test graphs in: {output_dir}")
    print(">>> FWHT curves (3 families) should nearly overlap (g-dependent)")
    print(">>> DFS curves should diverge sharply across families")
    print("=" * 60)
