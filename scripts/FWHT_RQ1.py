import networkx as nx
import random
import os
import shutil

def generate_figure1_dataset():
    """Figure 1 (unchanged): Execution time vs g at fixed m."""
    output_dir = r"D:\cpp\.vscode\FWHTRQ0GR"
    os.makedirs(output_dir, exist_ok=True)

    m_targets = [30, 40, 50]
    total_files = 0

    for m_target in m_targets:
        g_max = min(18, m_target - 2)

        for g in range(5, g_max + 1):
            n = m_target - g + 1

            if m_target > n * (n - 1) // 2:
                continue

            for i in range(1, 4):
                G = None
                while G is None:
                    seq = [2] * n
                    edges_needed = m_target - n

                    temp_seq = list(seq)
                    for _ in range(edges_needed):
                        valid_nodes = [v for v, d in enumerate(temp_seq) if d + 2 <= n - 1]
                        if not valid_nodes:
                            break
                        v = random.choice(valid_nodes)
                        temp_seq[v] += 2

                    if sum(temp_seq) == 2 * m_target and nx.is_graphical(temp_seq):
                        random.shuffle(temp_seq)
                        G_temp = nx.havel_hakimi_graph(temp_seq)
                        try:
                            nx.double_edge_swap(G_temp, nswap=5*m_target, max_tries=100*m_target)
                        except nx.NetworkXError:
                            pass
                        if nx.is_connected(G_temp):
                            G = G_temp

                mapping = list(range(n))
                random.shuffle(mapping)
                G = nx.relabel_nodes(G, {old: new for old, new in zip(range(n), mapping)})

                filename = os.path.join(output_dir, f"graph_m{m_target}_g{g}_{i}.gr")
                with open(filename, "w") as f:
                    f.write(f"p edge {n} {m_target}\n")
                    edges = list(G.edges())
                    random.shuffle(edges)
                    for u, v in edges:
                        if random.random() > 0.5:
                            f.write(f"e {u + 1} {v + 1}\n")
                        else:
                            f.write(f"e {v + 1} {u + 1}\n")

                total_files += 1

    print(f"Figure 1: {total_files} graphs generated -> {output_dir}")


def generate_figure2_dataset():
    """
    New Figure 2: Fix circuit rank g, vary m (and thus n = m - g + 1).

    At fixed g, increasing m adds nodes and edges while keeping the
    topological complexity constant. Smaller m means denser graphs
    (higher avg degree), larger m means sparser graphs (lower avg degree).

    Degree sequences are randomly generated (all even, min 2), so
    different instances at the same (g, m) will produce genuinely
    different graph structures with varying ec(G).
    """
    output_dir = r"D:\cpp\.vscode\FWHTRQ1_FIG2"
    # Clean old data to avoid mixing with previous k-regular files
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    total_files = 0

    # For each fixed g, a range of m values giving different densities
    # n = m - g + 1,  avg_degree = 2m/n
    experiments = {
        10: [18, 22, 27, 35, 50],   # avg_deg: 4.0, 3.4, 3.0, 2.7, 2.4
        12: [22, 27, 33, 40, 50],   # avg_deg: 4.0, 3.4, 3.0, 2.8, 2.6
        14: [26, 32, 40, 50],       # avg_deg: 4.0, 3.4, 3.0, 2.7
    }

    for g, m_list in experiments.items():
        for m_target in m_list:
            n = m_target - g + 1
            avg_deg = 2 * m_target / n

            # Feasibility: simple graph needs m <= n*(n-1)/2
            if m_target > n * (n - 1) // 2:
                print(f"  [SKIP] g={g}, m={m_target}: too dense for n={n}")
                continue

            print(f"  g={g}, m={m_target}, n={n}, avg_deg={avg_deg:.2f}")

            for i in range(1, 4):
                G = None
                attempts = 0
                while G is None and attempts < 1000:
                    attempts += 1
                    # Build random all-even degree sequence summing to 2*m_target
                    temp_seq = [2] * n
                    remaining = m_target - n  # each +2 to a node adds 1 edge

                    for _ in range(remaining):
                        valid = [v for v, d in enumerate(temp_seq) if d + 2 <= n - 1]
                        if not valid:
                            break
                        v = random.choice(valid)
                        temp_seq[v] += 2

                    if sum(temp_seq) != 2 * m_target:
                        continue
                    if not nx.is_graphical(temp_seq):
                        continue

                    random.shuffle(temp_seq)
                    try:
                        G_temp = nx.havel_hakimi_graph(temp_seq)
                    except Exception:
                        continue
                    try:
                        nx.double_edge_swap(G_temp, nswap=5*m_target, max_tries=100*m_target)
                    except nx.NetworkXError:
                        pass
                    if nx.is_connected(G_temp):
                        G = G_temp

                if G is None:
                    print(f"    [WARN] Failed: g={g}, m={m_target}, instance {i}")
                    continue

                # Randomize node labels and edge order
                mapping = list(range(n))
                random.shuffle(mapping)
                G = nx.relabel_nodes(G, {old: new for old, new in zip(range(n), mapping)})

                filename = os.path.join(output_dir, f"graph_g{g}_m{m_target}_{i}.gr")
                with open(filename, "w") as f:
                    f.write(f"p edge {n} {m_target}\n")
                    edges = list(G.edges())
                    random.shuffle(edges)
                    for u, v in edges:
                        if random.random() > 0.5:
                            f.write(f"e {u + 1} {v + 1}\n")
                        else:
                            f.write(f"e {v + 1} {u + 1}\n")

                total_files += 1

    print(f"\nFigure 2: {total_files} graphs generated -> {output_dir}")


if __name__ == "__main__":
    generate_figure1_dataset()
    generate_figure2_dataset()
