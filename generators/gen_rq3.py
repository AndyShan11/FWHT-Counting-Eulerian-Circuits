"""
RQ3 Graph Generator: Effect of Graph Size (M) at Fixed Circuit Rank (g).

Generates Eulerian graphs at FIXED circuit rank g with varying edge count M.
Three graph families:
  1. Cactus           — K cycles sharing only cut-vertices (g = K)
  2. Chain-of-Cycles  — cycles linked sequentially (g = K)
  3. Random-Augmented — tree + g random back-edges, parity-repaired

Output: DIMACS .gr format
"""

import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)


# ---------- Family 1: Cactus graph ----------
def generate_cactus(M, g):
    """
    Build a cactus with exactly g cycles whose total edge count is M.
    Each cycle contributes 1 to the circuit rank.  V = M - g + 1.
    """
    K = g
    if M < 3 * K:
        raise ValueError(f"M={M} too small for {K} cycles (need >= {3*K})")
    cycle_lengths = [3] * K
    remaining = M - 3 * K
    for _ in range(remaining):
        cycle_lengths[random.randint(0, K - 1)] += 1

    edges = []
    vertices = [1]
    cur_id = 1

    for length in cycle_lengths:
        attach = random.choice(vertices)
        new_nodes = list(range(cur_id + 1, cur_id + length))
        vertices.extend(new_nodes)
        cur_id += length - 1
        cycle_nodes = [attach] + new_nodes
        for i in range(length):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % length]
            edges.append((u, v))

    V = cur_id
    return V, M, edges


# ---------- Family 2: Chain of cycles ----------
def generate_chain_of_cycles(M, g):
    """
    Build g cycles linked sequentially (last node of cycle i = first of i+1).
    Circuit rank = g.
    """
    K = g
    if M < 3 * K:
        raise ValueError(f"M={M} too small for chain-of-{K}-cycles")
    cycle_lengths = [3] * K
    remaining = M - 3 * K
    for _ in range(remaining):
        cycle_lengths[random.randint(0, K - 1)] += 1

    edges = []
    cur_id = 1

    for ci, length in enumerate(cycle_lengths):
        start_node = cur_id
        nodes_in_cycle = [start_node + j for j in range(length)]
        for i in range(length):
            u = nodes_in_cycle[i]
            v = nodes_in_cycle[(i + 1) % length]
            edges.append((u, v))
        cur_id = nodes_in_cycle[-1]

    V = max(max(u, v) for u, v in edges)
    return V, len(edges), edges


# ---------- Family 3: Random Eulerian with controlled g ----------
def generate_random_eulerian(M, g):
    """
    Start from a path (tree) on n = M - g + 1 nodes, add g random
    back-edges, then ensure all degrees are even by pairing odd-degree
    nodes.  Retries until the actual circuit rank matches g.
    """
    n = M - g + 1
    if n < 3:
        raise ValueError(f"M={M}, g={g} yields n={n} < 3")

    max_tries = 50
    for _ in range(max_tries):
        edges = []
        for i in range(1, n):
            edges.append((i, i + 1))

        added = 0
        attempts = 0
        while added < g and attempts < g * 20:
            u = random.randint(1, n)
            v = random.randint(1, n)
            if u != v and abs(u - v) > 1:
                edges.append((u, v))
                added += 1
            attempts += 1
        if added < g:
            continue

        from collections import Counter
        def get_degrees(edge_list, num_v):
            deg = Counter()
            for u, v in edge_list:
                deg[u] += 1
                deg[v] += 1
            return deg

        deg = get_degrees(edges, n)
        odd_nodes = [v for v in range(1, n + 1) if deg[v] % 2 == 1]
        random.shuffle(odd_nodes)
        while len(odd_nodes) >= 2:
            a = odd_nodes.pop()
            b = odd_nodes.pop()
            edges.append((a, b))

        actual_m = len(edges)
        actual_g = actual_m - n + 1
        if actual_g == g:
            return n, actual_m, edges

    # Fallback to cactus if random generation fails
    return generate_cactus(M, g)


# ---------- I/O ----------
def save_gr(filename, V, E, edges):
    with open(filename, 'w') as f:
        f.write(f"c RQ3 Eulerian graph V={V} E={E}\n")
        f.write(f"p edge {V} {E}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")


# ======================== Main ========================
if __name__ == "__main__":
    # Fixed circuit ranks to test
    G_VALUES = [8, 10, 12]

    # For each g, increasing M values (M >= 3g for cactus feasibility)
    M_RANGES = {
        8:  [30, 50, 70, 90, 110, 130, 150, 180, 210, 250],
        10: [40, 60, 80, 100, 120, 150, 180, 210, 250, 300],
        12: [50, 70, 90, 120, 150, 180, 220, 260, 300, 350],
    }

    FAMILIES = {
        'cactus': generate_cactus,
        'chain':  generate_chain_of_cycles,
        'random': generate_random_eulerian,
    }

    REPEATS = 3  # instances per (family, g, M) for error bars

    output_dir = os.path.join(REPO_ROOT, "data", "rq3_graphs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Output directory: {output_dir}")
    print()
    print(f"{'Family':<10} {'g':<4} {'M_target':<10} {'V':<6} "
          f"{'M_actual':<10} {'file'}")
    print("-" * 70)

    for g in G_VALUES:
        for m_target in M_RANGES[g]:
            for fam_name, gen_fn in FAMILIES.items():
                for rep in range(1, REPEATS + 1):
                    try:
                        V, E, edges = gen_fn(m_target, g)
                        fname = f"{fam_name}_g{g}_M{E}_r{rep}.gr"
                        fpath = os.path.join(output_dir, fname)
                        save_gr(fpath, V, E, edges)
                        print(f"{fam_name:<10} {g:<4} {m_target:<10} "
                              f"{V:<6} {E:<10} {fname}")
                    except Exception as ex:
                        print(f"{fam_name:<10} {g:<4} {m_target:<10} "
                              f"SKIP ({ex})")
