"""
RQ4: Generate benchmark graphs at different scales for parallel scaling experiments.
Creates graphs with varying (g, M) to test how speedup changes with problem size.
"""
import random
import os

def generate_cactus(M, g):
    K = g
    if M < 3 * K:
        raise ValueError(f"M={M} too small for {K} cycles")
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

def save_gr(filename, V, E, edges):
    with open(filename, 'w') as f:
        f.write(f"c RQ4 benchmark V={V} E={E}\n")
        f.write(f"p edge {V} {E}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

if __name__ == "__main__":
    # Multiple problem sizes: (g, M) pairs
    # Small, medium, large to show how parallelism scales with problem size
    # M must be >= 3*g for cactus graphs
    configs = [
        (10, 40),   # small:  2^10 = 1024 states
        (12, 45),   # medium: 2^12 = 4096 states
        (13, 50),   # medium-large: 2^13 = 8192 states
        (14, 55),   # large:  2^14 = 16384 states
        (15, 60),   # very large: 2^15 = 32768 states
    ]

    output_dir = "RQ4_graphs"
    os.makedirs(output_dir, exist_ok=True)

    for g, M in configs:
        V, E, edges = generate_cactus(M, g)
        fname = f"rq4_g{g}_M{E}.gr"
        fpath = os.path.join(output_dir, fname)
        save_gr(fpath, V, E, edges)
        n = V
        print(f"Generated: {fname}  (V={V}, E={E}, g={g})")
