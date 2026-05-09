"""
Select representative graphs and extract infrastructure subgraphs.
Produces a curated set of ~30-50 .gr files for the paper's RQ5 (real-world experiments).
"""

import os
import re
from collections import defaultdict
from pathlib import Path

CONVERTED_DIR = Path(__file__).parent / "converted_gr"
SELECTED_DIR = Path(__file__).parent / "selected_gr"
SELECTED_DIR.mkdir(exist_ok=True)

MIN_G = 3
MAX_G = 18


def parse_gr_header(filepath):
    n, m, cr = 0, 0, 0
    augmented = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("c"):
                if "circuit_rank=" in line:
                    cr = int(re.search(r"circuit_rank=(\d+)", line).group(1))
                if "Augmented" in line:
                    augmented = True
            elif line.startswith("p"):
                parts = line.split()
                n = int(parts[2])
            elif line.startswith("e"):
                m += 1
    return {"n": n, "m": m, "cr": cr, "augmented": augmented, "path": filepath}


def select_from_dataset(dataset_name, max_per_cr=2, max_total=15):
    """Select representative graphs spanning circuit rank range."""
    ds_dir = CONVERTED_DIR / dataset_name
    if not ds_dir.exists():
        return []

    all_graphs = []
    for f in sorted(ds_dir.glob("*.gr")):
        info = parse_gr_header(str(f))
        info["name"] = f.stem
        all_graphs.append(info)

    by_cr = defaultdict(list)
    for g in all_graphs:
        by_cr[g["cr"]].append(g)

    selected = []
    for cr in sorted(by_cr.keys()):
        if cr < MIN_G or cr > MAX_G:
            continue
        candidates = sorted(by_cr[cr], key=lambda x: x["m"])
        for c in candidates[:max_per_cr]:
            selected.append(c)
        if len(selected) >= max_total:
            break

    return selected[:max_total]


def extract_induced_subgraphs(filepath, name, format_type):
    """Extract induced subgraphs from dense networks by taking k-hop neighborhoods."""
    from collections import deque

    edges_set = set()
    nodes = set()

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("%") or line.startswith("#") or not line:
                continue
            parts = line.split()
            if format_type == "mtx":
                if line.startswith("%%"):
                    continue
                if len(parts) == 3 and not any(c.isalpha() for c in line):
                    u, v = int(parts[0]), int(parts[1])
                elif len(parts) == 2:
                    u, v = int(parts[0]), int(parts[1])
                else:
                    continue
            else:
                if len(parts) < 2:
                    continue
                u, v = int(parts[0]), int(parts[1])

            if u != v:
                edges_set.add((min(u, v), max(u, v)))
                nodes.update([u, v])

    adj = defaultdict(set)
    deg = defaultdict(int)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    high_deg_nodes = sorted(deg.keys(), key=lambda x: -deg[x])

    results = []
    seen_sizes = set()

    for seed in high_deg_nodes[:30]:
        for hops in [1, 2]:
            visited = {seed}
            frontier = {seed}
            for _ in range(hops):
                next_frontier = set()
                for node in frontier:
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            next_frontier.add(nb)
                frontier = next_frontier

            sub_edges = set()
            for u in visited:
                for v in adj[u]:
                    if v in visited:
                        sub_edges.add((min(u, v), max(u, v)))

            sub_n = len(visited)
            sub_m = len(sub_edges)
            sub_g = sub_m - sub_n + 1

            size_key = (sub_n, sub_m)
            if size_key in seen_sizes:
                continue

            if MIN_G <= sub_g <= MAX_G:
                seen_sizes.add(size_key)
                node_map = {old: new + 1 for new, old in enumerate(sorted(visited))}
                mapped_edges = [(node_map[u], node_map[v]) for u, v in sub_edges]

                sub_deg = defaultdict(int)
                for u, v in mapped_edges:
                    sub_deg[u] += 1
                    sub_deg[v] += 1
                odd_nodes = sorted([v for v in sub_deg if sub_deg[v] % 2 == 1])

                edge_exists = set((min(u,v), max(u,v)) for u,v in mapped_edges)
                added = []
                cur_n = sub_n
                for i in range(0, len(odd_nodes) - 1, 2):
                    u, v = odd_nodes[i], odd_nodes[i + 1]
                    e = (min(u, v), max(u, v))
                    if e in edge_exists:
                        cur_n += 1
                        w = cur_n
                        added.append((min(u, w), max(u, w)))
                        added.append((min(v, w), max(v, w)))
                        edge_exists.add((min(u, w), max(u, w)))
                        edge_exists.add((min(v, w), max(v, w)))
                    else:
                        added.append(e)
                        edge_exists.add(e)

                final_edges = mapped_edges + added
                final_m = len(final_edges)
                final_g = final_m - cur_n + 1

                if MIN_G <= final_g <= MAX_G:
                    results.append({
                        "n": cur_n,
                        "m": final_m,
                        "cr": final_g,
                        "edges": final_edges,
                        "seed": seed,
                        "hops": hops,
                        "added": len(added),
                    })

        if len(results) >= 8:
            break

    return results


def write_gr(filepath, n, edges, comment=""):
    with open(filepath, "w") as f:
        if comment:
            for line in comment.split("\n"):
                f.write(f"c {line}\n")
        f.write(f"p sp {n}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")


def main():
    base_dir = Path(__file__).parent
    total = 0

    for ds_name in ["MUTAG", "AIDS", "PTC_MR"]:
        selected = select_from_dataset(ds_name)
        if not selected:
            continue
        out_dir = SELECTED_DIR / ds_name
        out_dir.mkdir(exist_ok=True)

        print(f"\n{ds_name}: selected {len(selected)} graphs")
        for g in selected:
            src = g["path"]
            dst = out_dir / Path(src).name
            with open(src) as f_in:
                content = f_in.read()
            with open(dst, "w") as f_out:
                f_out.write(content)
            print(f"  cr={g['cr']:2d}  n={g['n']:3d}  m={g['m']:3d}  {Path(src).name}")
            total += 1

    infra_dir = base_dir / "infrastructure"
    infra_out = SELECTED_DIR / "infrastructure"
    infra_out.mkdir(exist_ok=True)

    infra_files = [
        ("inf-USAir97.mtx", "inf-USAir97", "mtx"),
        ("inf-power.mtx", "inf-power", "mtx"),
        ("inf-euroroad.edges", "inf-euroroad", "edges"),
    ]

    for fname, name, fmt in infra_files:
        fpath = infra_dir / fname
        if not fpath.exists():
            continue
        subgraphs = extract_induced_subgraphs(str(fpath), name, fmt)
        if not subgraphs:
            print(f"\n{name}: no suitable subgraphs found")
            continue

        print(f"\n{name}: extracted {len(subgraphs)} subgraphs")
        for i, sg in enumerate(subgraphs):
            comment = (f"Source: {name} ({sg['hops']}-hop neighborhood of node {sg['seed']})\n"
                      f"n={sg['n']}, m={sg['m']}, circuit_rank={sg['cr']}\n"
                      f"Augmented with {sg['added']} edges for Eulerian condition")
            out_name = f"{name}_sub{i}_n{sg['n']}_m{sg['m']}_cr{sg['cr']}.gr"
            write_gr(infra_out / out_name, sg["n"], sg["edges"], comment)
            print(f"  cr={sg['cr']:2d}  n={sg['n']:3d}  m={sg['m']:3d}  {out_name}")
            total += 1

    print(f"\n{'='*60}")
    print(f"Total selected .gr files: {total}")
    print(f"Output directory: {SELECTED_DIR}")


if __name__ == "__main__":
    main()
