"""
Convert real-world graph datasets to DIMACS .gr format for FWHT Euler circuit counting.

Handles:
1. TUDatasets (MUTAG, AIDS, PTC_MR) - molecular graphs
2. Network Repository (.edges, .mtx) - infrastructure networks

For each graph:
- Compute circuit rank g = m - n + 1
- Check if Eulerian (all vertices even degree)
- If not Eulerian, augment with minimum edges to make Eulerian
- Filter for target circuit rank range
- Output in DIMACS .gr format
"""

import os
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "converted_gr"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_G = 3
MAX_G = 18


def parse_tu_dataset(dataset_dir, dataset_name):
    base = Path(dataset_dir) / dataset_name / dataset_name
    a_file = base / f"{dataset_name}_A.txt"
    ind_file = base / f"{dataset_name}_graph_indicator.txt"

    edges = []
    with open(a_file) as f:
        for line in f:
            parts = line.strip().split(",")
            u, v = int(parts[0].strip()), int(parts[1].strip())
            edges.append((u, v))

    indicators = []
    with open(ind_file) as f:
        for line in f:
            indicators.append(int(line.strip()))

    num_graphs = max(indicators)
    graphs = {}
    for gid in range(1, num_graphs + 1):
        graphs[gid] = {"nodes": set(), "edges": set()}

    for node_idx, gid in enumerate(indicators, 1):
        graphs[gid]["nodes"].add(node_idx)

    for u, v in edges:
        if u < v:
            gid = indicators[u - 1]
            graphs[gid]["edges"].add((u, v))

    result = []
    for gid in range(1, num_graphs + 1):
        g = graphs[gid]
        nodes = sorted(g["nodes"])
        node_map = {old: new + 1 for new, old in enumerate(nodes)}
        mapped_edges = [(node_map[u], node_map[v]) for u, v in g["edges"]]
        result.append({"id": gid, "n": len(nodes), "edges": mapped_edges})
    return result


def parse_edges_file(filepath):
    edges = set()
    nodes = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("%") or line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edges.add((min(u, v), max(u, v)))
                    nodes.update([u, v])
    node_map = {old: new + 1 for new, old in enumerate(sorted(nodes))}
    mapped_edges = [(node_map[u], node_map[v]) for u, v in edges]
    return [{"id": 1, "n": len(nodes), "edges": mapped_edges}]


def parse_mtx_file(filepath):
    edges = set()
    nodes = set()
    header_done = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                continue
            parts = line.split()
            if not header_done:
                header_done = True
                continue
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edges.add((min(u, v), max(u, v)))
                    nodes.update([u, v])
    node_map = {old: new + 1 for new, old in enumerate(sorted(nodes))}
    mapped_edges = [(node_map[u], node_map[v]) for u, v in edges]
    return [{"id": 1, "n": len(nodes), "edges": mapped_edges}]


def compute_degree(n, edges):
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def is_connected(n, edges):
    if n == 0:
        return False
    adj = defaultdict(set)
    all_nodes = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        all_nodes.update([u, v])
    if not all_nodes:
        return False
    start = next(iter(all_nodes))
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(all_nodes)


def make_eulerian(n, edges):
    """Add minimum edges to make the graph Eulerian, avoiding parallel edges.
    When two odd-degree nodes are already adjacent, insert a subdivision vertex.
    Returns (new_edge_list, new_n, num_added_edges).
    """
    edge_set = set(edges)
    deg = compute_degree(n, edges)
    all_nodes = set()
    for u, v in edges:
        all_nodes.update([u, v])
    odd_nodes = [v for v in all_nodes if deg[v] % 2 == 1]
    if not odd_nodes:
        return list(edges), n, 0

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    added = []
    new_n = max(all_nodes) if all_nodes else n
    odd_set = set(odd_nodes)

    while odd_set:
        u = min(odd_set)
        best_v = None
        visited = {u}
        queue = [(u, 0)]
        qi = 0
        while qi < len(queue):
            node, dist = queue[qi]
            qi += 1
            if node != u and node in odd_set:
                best_v = node
                break
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

        if best_v is None:
            best_v = min(odd_set - {u})

        e = (min(u, best_v), max(u, best_v))
        if e in edge_set:
            new_n += 1
            w = new_n
            added.append((min(u, w), max(u, w)))
            added.append((min(best_v, w), max(best_v, w)))
            edge_set.add((min(u, w), max(u, w)))
            edge_set.add((min(best_v, w), max(best_v, w)))
            adj[u].add(w)
            adj[w].add(u)
            adj[best_v].add(w)
            adj[w].add(best_v)
        else:
            added.append(e)
            edge_set.add(e)
            adj[u].add(best_v)
            adj[best_v].add(u)

        odd_set.discard(u)
        odd_set.discard(best_v)

    return list(edges) + added, new_n, len(added)


def extract_subgraph_with_target_g(n, edges, target_g_range=(MIN_G, MAX_G)):
    deg = compute_degree(n, edges)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    sorted_nodes = sorted(deg.keys(), key=lambda x: -deg[x])
    results = []

    for seed_idx in range(min(10, len(sorted_nodes))):
        seed = sorted_nodes[seed_idx]
        visited_nodes = {seed}
        visited_edges = set()
        queue = [seed]
        qi = 0

        while qi < len(queue):
            node = queue[qi]
            qi += 1
            for nb in adj[node]:
                e = (min(node, nb), max(node, nb))
                if e not in visited_edges:
                    visited_edges.add(e)
                    if nb not in visited_nodes:
                        visited_nodes.add(nb)
                        queue.append(nb)
                current_g = len(visited_edges) - len(visited_nodes) + 1
                if current_g >= target_g_range[1]:
                    break
            current_g = len(visited_edges) - len(visited_nodes) + 1
            if current_g >= target_g_range[1]:
                break

        current_g = len(visited_edges) - len(visited_nodes) + 1
        if target_g_range[0] <= current_g <= target_g_range[1]:
            node_map = {old: new + 1 for new, old in enumerate(sorted(visited_nodes))}
            mapped_edges = [(node_map[u], node_map[v]) for u, v in visited_edges]
            results.append({
                "n": len(visited_nodes),
                "edges": mapped_edges,
                "g": current_g,
                "seed": seed,
            })
    return results


def write_gr(filepath, n, edges, comment=""):
    with open(filepath, "w") as f:
        if comment:
            for line in comment.split("\n"):
                f.write(f"c {line}\n")
        f.write(f"p sp {n}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")


def process_tu_dataset(dataset_dir, dataset_name):
    print(f"\n{'='*60}")
    print(f"Processing TUDataset: {dataset_name}")
    print(f"{'='*60}")

    graphs = parse_tu_dataset(dataset_dir, dataset_name)
    print(f"  Total graphs: {len(graphs)}")

    stats = {"total": len(graphs), "eulerian": 0, "augmented": 0, "output": 0}
    out_dir = OUTPUT_DIR / dataset_name
    out_dir.mkdir(exist_ok=True)

    for g_data in graphs:
        gid = g_data["id"]
        n = g_data["n"]
        edges = g_data["edges"]
        m = len(edges)

        if n < 3 or m < 3:
            continue
        if not is_connected(n, edges):
            continue

        circuit_rank = m - n + 1
        deg = compute_degree(n, edges)
        all_even = all(d % 2 == 0 for d in deg.values())

        if all_even and MIN_G <= circuit_rank <= MAX_G:
            stats["eulerian"] += 1
            comment = (f"Source: TUDataset {dataset_name}, graph #{gid}\n"
                      f"n={n}, m={m}, circuit_rank={circuit_rank}\n"
                      f"Naturally Eulerian")
            fname = f"{dataset_name}_g{gid}_n{n}_m{m}_cr{circuit_rank}_natural.gr"
            write_gr(out_dir / fname, n, edges, comment)
            stats["output"] += 1
        elif not all_even:
            aug_edges, new_n, num_added = make_eulerian(n, edges)
            new_m = len(aug_edges)
            new_cr = new_m - new_n + 1
            if MIN_G <= new_cr <= MAX_G:
                stats["augmented"] += 1
                comment = (f"Source: TUDataset {dataset_name}, graph #{gid}\n"
                          f"Original: n={n}, m={m}\n"
                          f"Augmented: +{num_added} edges, n={new_n} -> m={new_m}, circuit_rank={new_cr}\n"
                          f"Augmented to satisfy Eulerian condition (no parallel edges)")
                fname = f"{dataset_name}_g{gid}_n{new_n}_m{new_m}_cr{new_cr}_augmented.gr"
                write_gr(out_dir / fname, new_n, aug_edges, comment)
                stats["output"] += 1

    print(f"  Naturally Eulerian with {MIN_G}<=g<={MAX_G}: {stats['eulerian']}")
    print(f"  Augmented to Eulerian with {MIN_G}<=g<={MAX_G}: {stats['augmented']}")
    print(f"  Total output: {stats['output']}")
    return stats


def process_infrastructure(filepath, name):
    print(f"\n{'='*60}")
    print(f"Processing infrastructure: {name}")
    print(f"{'='*60}")

    if filepath.endswith(".edges"):
        graphs = parse_edges_file(filepath)
    elif filepath.endswith(".mtx"):
        graphs = parse_mtx_file(filepath)
    else:
        print(f"  Unknown format: {filepath}")
        return

    g_data = graphs[0]
    n = g_data["n"]
    m = len(g_data["edges"])
    circuit_rank = m - n + 1
    print(f"  Full graph: n={n}, m={m}, circuit_rank={circuit_rank}")

    out_dir = OUTPUT_DIR / "infrastructure"
    out_dir.mkdir(exist_ok=True)

    if MIN_G <= circuit_rank <= MAX_G:
        edges = g_data["edges"]
        aug_edges, new_n, num_added = make_eulerian(n, edges)
        new_m = len(aug_edges)
        new_cr = new_m - new_n + 1
        comment = (f"Source: {name}\n"
                  f"n={new_n}, m={new_m}, circuit_rank={new_cr}\n"
                  f"Augmented with {num_added} edges for Eulerian condition")
        fname = f"{name}_n{new_n}_m{new_m}_cr{new_cr}.gr"
        write_gr(out_dir / fname, new_n, aug_edges, comment)
        print(f"  Output: {fname}")
    else:
        print(f"  Circuit rank {circuit_rank} out of range, extracting subgraphs...")
        subgraphs = extract_subgraph_with_target_g(n, g_data["edges"])
        for i, sg in enumerate(subgraphs):
            sg_edges = sg["edges"]
            sg_n = sg["n"]
            aug_edges, new_sg_n, num_added = make_eulerian(sg_n, sg_edges)
            new_m = len(aug_edges)
            new_cr = new_m - new_sg_n + 1
            if MIN_G <= new_cr <= MAX_G:
                comment = (f"Source: {name} (subgraph around node {sg['seed']})\n"
                          f"Subgraph: n={sg_n}, original_m={len(sg_edges)}, g={sg['g']}\n"
                          f"Augmented: +{num_added} edges, n={new_sg_n} -> m={new_m}, circuit_rank={new_cr}")
                fname = f"{name}_sub{i}_n{new_sg_n}_m{new_m}_cr{new_cr}.gr"
                write_gr(out_dir / fname, new_sg_n, aug_edges, comment)
                print(f"  Output: {fname}")


def main():
    base_dir = Path(__file__).parent

    mol_dir = base_dir / "molecular"
    for ds_name in ["MUTAG", "AIDS", "PTC_MR"]:
        ds_path = mol_dir / ds_name
        if ds_path.exists():
            process_tu_dataset(str(mol_dir), ds_name)

    infra_dir = base_dir / "infrastructure"
    infra_files = [
        (str(infra_dir / "inf-euroroad.edges"), "inf-euroroad"),
        (str(infra_dir / "inf-power.mtx"), "inf-power"),
        (str(infra_dir / "inf-USAir97.mtx"), "inf-USAir97"),
    ]
    for fpath, name in infra_files:
        if os.path.exists(fpath):
            process_infrastructure(fpath, name)

    print(f"\n{'='*60}")
    print(f"All converted files in: {OUTPUT_DIR}")
    total = sum(1 for _ in OUTPUT_DIR.rglob("*.gr"))
    print(f"Total .gr files: {total}")


if __name__ == "__main__":
    main()
