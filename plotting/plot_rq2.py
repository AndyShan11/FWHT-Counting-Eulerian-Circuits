"""
RQ2 Plotting: FWHT vs DFS Comparison (Sliding-N Strategy).

Generates 4 publication-quality figures:
  Fig 1: FWHT vs DFS Runtime vs Genus (main result)
  Fig 2: Speedup (DFS / FWHT) vs Genus
  Fig 3: Runtime vs Edge Count (m)
  Fig 4: Complete Data Table with Speedup & TIMEOUT highlight

FWHT data is embedded (from solver runs); DFS data is read from CSV.
On DFS timeout, the plotting script extrapolates total DFS time using
partial progress and known FWHT circuit counts.

Dependencies: numpy, matplotlib, csv
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import csv
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FIG_DIR    = os.path.join(REPO_ROOT, "figures", "rq2")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# 1. Raw Data  (FWHT from fwht_solver runs, DFS from dfs_solver CSV)
# ============================================================================

# FWHT results: filename -> (N, M, genus, time_s, circuits)
fwht_data = {
    "sparse_n15_g5.gr":  (15, 20, 6,  0.0025,   112),
    "sparse_n16_g6.gr":  (16, 21, 6,  0.0034,   120),
    "sparse_n17_g7.gr":  (17, 24, 8,  0.0121,   800),
    "sparse_n18_g8.gr":  (18, 27, 10, 0.0964,   13984),
    "sparse_n19_g9.gr":  (19, 29, 11, 0.2809,   32640),
    "sparse_n20_g10.gr": (20, 31, 12, 0.8010,   100416),
    "sparse_n22_g11.gr": (22, 33, 12, 0.6924,   53984),
    "sparse_n24_g12.gr": (24, 37, 14, 4.5130,   2243584),
    "sparse_n27_g13.gr": (27, 40, 14, 5.3168,   1355008),
    "sparse_n30_g14.gr": (30, 45, 16, 59.5978,  14532896),
    "medium_n8_g6.gr":   (8,  15, 8,  0.0073,   2528),
    "medium_n8_g7.gr":   (8,  15, 8,  0.0111,   1712),
    "medium_n9_g8.gr":   (9,  19, 11, 0.0860,   86016),
    "medium_n9_g9.gr":   (9,  20, 12, 0.1425,   277248),
    "medium_n10_g10.gr": (10, 21, 12, 0.1870,   359424),
    "medium_n10_g11.gr": (10, 23, 14, 1.2739,   5661696),
    "medium_n10_g12.gr": (10, 24, 15, 2.6477,   195600384),
    "dense_n7_g6.gr":    (7,  13, 7,  0.0026,   592),
    "dense_n7_g7.gr":    (7,  16, 10, 0.0282,   61568),
    "dense_n7_g8.gr":    (7,  16, 10, 0.0240,   47808),
    "dense_n6_g9.gr":    (6,  18, 13, 0.2740,   7681536),
    "dense_n6_g10.gr":   (6,  18, 13, 0.2439,   7326720),
    "tangled_n7_g10.gr": (7,  20, 14, 1.8129,   23344128),
    "tangled_n7_g11.gr": (7,  21, 15, 4.6241,   489673728),
    "tangled_n7_g12.gr": (7,  21, 15, 4.7140,   156556800),
    "tangled_n7_g13.gr": (7,  24, 18, 41.7678,  31950360576),
    "tangled_n7_g14.gr": (7,  26, 20, 252.503,  4580543250432),
    "tangled_n7_g15.gr": (7,  27, 21, 419.905,  76228289740800),
}

# DFS results from CSV
dfs_csv_path = os.path.join(REPO_ROOT, "results", "eulerian_RQ2_DFS_results.csv")
dfs_data = {}
if os.path.exists(dfs_csv_path):
    with open(dfs_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["Filename"].strip()
            dfs_data[fn] = (
                row["Status"].strip(),
                int(row["Circuits"]),
                float(row["Time(s)"]),
                int(row["PartialRaw"]),
            )
else:
    print(f"[WARN] DFS CSV not found: {dfs_csv_path}")
    print("       Run dfs_solver first, then re-run this script.")

# ============================================================================
# 2. Merge & Classify
# ============================================================================

def get_family(fn):
    if fn.startswith("sparse"):  return "Sparse"
    if fn.startswith("medium"):  return "Medium"
    if fn.startswith("dense"):   return "Dense"
    if fn.startswith("tangled"): return "Tangled"
    return "Unknown"

def get_target_g(fn):
    parts = fn.replace(".gr", "").split("_")
    for p in parts:
        if p.startswith("g"):
            return int(p[1:])
    return 0

records = []
for fn in sorted(fwht_data.keys()):
    if fn not in dfs_data:
        continue
    family = get_family(fn)
    target_g = get_target_g(fn)
    n, m, genus, fwht_t, fwht_c = fwht_data[fn]
    dfs_status, dfs_c, dfs_t, dfs_raw = dfs_data[fn]
    records.append({
        "file": fn, "family": family, "target_g": target_g,
        "N": n, "M": m, "genus": genus,
        "fwht_time": fwht_t, "fwht_circuits": fwht_c,
        "dfs_status": dfs_status, "dfs_time": dfs_t,
        "dfs_circuits": dfs_c, "dfs_raw": dfs_raw,
    })

# ============================================================================
# 3. Extrapolate TIMEOUT entries
# ============================================================================
family_ratios = defaultdict(list)
for r in records:
    if (r["dfs_status"] == "OK" and r["fwht_circuits"] > 0
            and r["dfs_circuits"] > 0):
        total_raw = r["dfs_circuits"] * 2 * r["M"]
        ratio = total_raw / r["fwht_circuits"]
        family_ratios[r["family"]].append(ratio)

family_avg_ratio = {}
for fam, ratios in family_ratios.items():
    family_avg_ratio[fam] = np.median(ratios) if ratios else 1.0
    print(f"  [Ratio] {fam}: median(dfs_raw / fwht_circuits) = "
          f"{family_avg_ratio[fam]:.3f}")

for r in records:
    if r["dfs_status"] == "TIMEOUT":
        ratio = family_avg_ratio.get(r["family"], 1.0)
        estimated_total_raw = r["fwht_circuits"] * ratio
        rate = r["dfs_raw"] / 600.0
        if rate > 0:
            r["dfs_extrapolated"] = estimated_total_raw / rate
        else:
            r["dfs_extrapolated"] = float("inf")
        print(f"  [Extrap] {r['file']}: est. DFS = "
              f"{r['dfs_extrapolated']:.1f}s "
              f"({r['dfs_extrapolated']/3600:.1f}h)")
    else:
        r["dfs_extrapolated"] = r["dfs_time"]

# ============================================================================
# 4. Plotting Configuration
# ============================================================================
FAMILY_STYLE = {
    "Sparse":  {"color": "#2196F3", "marker": "s",
                "label": "Sparse (N=15\u201330)"},
    "Medium":  {"color": "#FF9800", "marker": "o",
                "label": "Medium (N=8\u201310)"},
    "Dense":   {"color": "#E91E63", "marker": "D",
                "label": "Dense (N=6\u20137)"},
    "Tangled": {"color": "#9C27B0", "marker": "^",
                "label": "Tangled (N=7)"},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

# ============================================================================
# Figure 1: FWHT vs DFS Runtime vs Genus
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7))

for fam in ["Sparse", "Medium", "Dense", "Tangled"]:
    sty = FAMILY_STYLE[fam]
    sub = sorted([r for r in records if r["family"] == fam],
                 key=lambda x: x["genus"])
    if not sub:
        continue

    g_vals    = [r["genus"] for r in sub]
    fwht_vals = [r["fwht_time"] for r in sub]
    dfs_ok_g  = [r["genus"]    for r in sub if r["dfs_status"] == "OK"]
    dfs_ok_t  = [r["dfs_time"] for r in sub if r["dfs_status"] == "OK"]
    dfs_to_g  = [r["genus"]           for r in sub
                 if r["dfs_status"] == "TIMEOUT"]
    dfs_to_t  = [r["dfs_extrapolated"] for r in sub
                 if r["dfs_status"] == "TIMEOUT"]

    # FWHT: solid line with filled markers
    ax1.plot(g_vals, fwht_vals, color=sty["color"], marker=sty["marker"],
             markersize=8, linewidth=2, linestyle="-", zorder=5,
             label=f"FWHT - {sty['label']}")

    # DFS OK: dashed line with open markers
    if dfs_ok_g:
        ax1.plot(dfs_ok_g, dfs_ok_t, color=sty["color"],
                 marker=sty["marker"], markersize=8, linewidth=2,
                 linestyle="--", alpha=0.85, zorder=4,
                 markerfacecolor="none", markeredgewidth=2)

    # DFS TIMEOUT (extrapolated)
    if dfs_to_g:
        ax1.scatter(dfs_to_g, dfs_to_t, color=sty["color"], marker="X",
                    s=160, zorder=6, edgecolors="black", linewidths=1.0)
        for gi, ti in zip(dfs_to_g, dfs_to_t):
            ax1.annotate("", xy=(gi, ti), xytext=(gi, 600),
                         arrowprops=dict(arrowstyle="->", color=sty["color"],
                                         lw=1.5, linestyle=":", alpha=0.5))
        if dfs_ok_g:
            ax1.plot([dfs_ok_g[-1], dfs_to_g[0]],
                     [dfs_ok_t[-1], dfs_to_t[0]],
                     color=sty["color"], linestyle=":", linewidth=1.2,
                     alpha=0.4)

# 600s timeout line
ax1.axhline(y=600, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
ax1.text(21.5, 800, "DFS Timeout = 600 s", color="red",
         ha="right", fontsize=9, fontstyle="italic")

ax1.set_yscale("log")
ax1.set_xlabel("Graph Genus ($g$)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Computation Time (s, log scale)", fontsize=13,
               fontweight="bold")
ax1.set_title("RQ2: FWHT vs DFS \u2014 Runtime vs Graph Genus\n"
              "(Solid = FWHT, Dashed = DFS, "
              "\u2715 = DFS Extrapolated from Timeout)",
              fontsize=13, fontweight="bold")

# Custom legend
legend_el = []
for fam in ["Sparse", "Medium", "Dense", "Tangled"]:
    sty = FAMILY_STYLE[fam]
    legend_el.append(Line2D(
        [0], [0], color=sty["color"], marker=sty["marker"],
        markersize=7, linewidth=2, linestyle="-",
        label=f"FWHT \u2014 {sty['label']}"))
    legend_el.append(Line2D(
        [0], [0], color=sty["color"], marker=sty["marker"],
        markersize=7, linewidth=2, linestyle="--",
        markerfacecolor="none", markeredgewidth=2,
        label=f"DFS  \u2014 {sty['label']}"))
legend_el.append(Line2D(
    [0], [0], color="black", marker="X", markersize=10,
    linestyle="None", label="DFS Timeout (extrapolated)"))
legend_el.append(Line2D(
    [0], [0], color="red", linestyle=":", linewidth=1.5,
    label="600 s Timeout Limit"))

ax1.legend(handles=legend_el, loc="upper left", fontsize=7.5, ncol=2,
           framealpha=0.9, edgecolor="gray")
fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "RQ2_fig1_fwht_vs_dfs_genus.png"),
             dpi=300, bbox_inches="tight")
print("[Saved] Fig 1")

# ============================================================================
# Figure 2: Speedup (DFS / FWHT) vs Genus
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(11, 7))

for fam in ["Sparse", "Medium", "Dense", "Tangled"]:
    sty = FAMILY_STYLE[fam]
    sub = sorted([r for r in records if r["family"] == fam],
                 key=lambda x: x["genus"])
    if not sub:
        continue

    g_ok, s_ok, g_ex, s_ex = [], [], [], []
    for r in sub:
        sp = (r["dfs_extrapolated"] / r["fwht_time"]
              if r["fwht_time"] > 0 else 0)
        if sp <= 0:
            continue
        if r["dfs_status"] == "OK":
            g_ok.append(r["genus"]); s_ok.append(sp)
        else:
            g_ex.append(r["genus"]); s_ex.append(sp)

    if g_ok:
        ax2.plot(g_ok, s_ok, color=sty["color"], marker=sty["marker"],
                 markersize=9, linewidth=2.5, linestyle="-", zorder=5,
                 label=sty["label"])
    if g_ex:
        ax2.scatter(g_ex, s_ex, color=sty["color"], marker="X", s=180,
                    zorder=6, edgecolors="black", linewidths=1.2)
        if g_ok:
            ax2.plot([g_ok[-1]] + g_ex, [s_ok[-1]] + s_ex,
                     color=sty["color"], linestyle=":", linewidth=1.5,
                     alpha=0.5)

# Break-even line
ax2.axhline(y=1, color="gray", linestyle="-", linewidth=1, alpha=0.5)
ax2.text(6.5, 1.4, "Break-even (DFS = FWHT)", fontsize=9, color="gray",
         fontstyle="italic")

ax2.set_yscale("log")
ax2.set_xlabel("Graph Genus ($g$)", fontsize=13, fontweight="bold")
ax2.set_ylabel("Speedup  (DFS Time / FWHT Time)", fontsize=13,
               fontweight="bold")
ax2.set_title("RQ2: FWHT Speedup over DFS\n"
              "(\u2715 = Extrapolated from 600 s Timeout)",
              fontsize=14, fontweight="bold")

# Annotate maximum speedup
all_sp = [(r["genus"], r["dfs_extrapolated"] / r["fwht_time"],
           r["family"], r["file"])
          for r in records
          if r["fwht_time"] > 0 and r["dfs_extrapolated"] > 0]
if all_sp:
    mx = max(all_sp, key=lambda x: x[1])
    ax2.annotate(f"Max: {mx[1]:,.0f}\u00d7\n({mx[2]}, g={mx[0]})",
                 xy=(mx[0], mx[1]),
                 xytext=(mx[0] - 3, mx[1] * 0.15),
                 fontsize=10, fontweight="bold", color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))

legend_el2 = []
for fam in ["Sparse", "Medium", "Dense", "Tangled"]:
    sty = FAMILY_STYLE[fam]
    legend_el2.append(Line2D(
        [0], [0], color=sty["color"], marker=sty["marker"],
        markersize=8, linewidth=2.5, label=sty["label"]))
legend_el2.append(Line2D(
    [0], [0], color="black", marker="X", markersize=10,
    linestyle="None", label="Extrapolated"))
ax2.legend(handles=legend_el2, loc="upper left", fontsize=10,
           framealpha=0.9)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "RQ2_fig2_speedup.png"),
             dpi=300, bbox_inches="tight")
print("[Saved] Fig 2")

# ============================================================================
# Figure 3: Runtime vs Edge Count (m)
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(12, 7))

for fam in ["Sparse", "Medium", "Dense", "Tangled"]:
    sty = FAMILY_STYLE[fam]
    sub = sorted([r for r in records if r["family"] == fam],
                 key=lambda x: x["M"])
    if not sub:
        continue

    m_vals    = [r["M"] for r in sub]
    fwht_vals = [r["fwht_time"] for r in sub]
    ax3.plot(m_vals, fwht_vals, color=sty["color"], marker=sty["marker"],
             markersize=8, linewidth=2, linestyle="-", zorder=5)

    m_ok = [r["M"]        for r in sub if r["dfs_status"] == "OK"]
    t_ok = [r["dfs_time"] for r in sub if r["dfs_status"] == "OK"]
    if m_ok:
        ax3.plot(m_ok, t_ok, color=sty["color"], marker=sty["marker"],
                 markersize=8, linewidth=2, linestyle="--", alpha=0.85,
                 zorder=4, markerfacecolor="none", markeredgewidth=2)

    m_to = [r["M"]               for r in sub
            if r["dfs_status"] == "TIMEOUT"]
    t_to = [r["dfs_extrapolated"] for r in sub
            if r["dfs_status"] == "TIMEOUT"]
    if m_to:
        ax3.scatter(m_to, t_to, color=sty["color"], marker="X", s=150,
                    zorder=6, edgecolors="black", linewidths=1.0)

ax3.axhline(y=600, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
ax3.text(45, 800, "DFS Timeout = 600 s", color="red", ha="right",
         fontsize=9, fontstyle="italic")

ax3.set_yscale("log")
ax3.set_xlabel("Number of Edges ($m$)", fontsize=13, fontweight="bold")
ax3.set_ylabel("Computation Time (s, log scale)", fontsize=13,
               fontweight="bold")
ax3.set_title("RQ2: FWHT vs DFS \u2014 Runtime vs Edge Count\n"
              "(Solid = FWHT, Dashed = DFS, \u2715 = DFS Extrapolated)",
              fontsize=13, fontweight="bold")

ax3.legend(handles=legend_el, loc="upper left", fontsize=7.5, ncol=2,
           framealpha=0.9, edgecolor="gray")
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "RQ2_fig3_time_vs_edges.png"),
             dpi=300, bbox_inches="tight")
print("[Saved] Fig 3")

# ============================================================================
# Figure 4: Complete Data Table
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(18, max(8, len(records) * 0.38 + 2)))
ax4.axis("off")

col_labels = ["Family", "File", "N", "M", "g",
              "FWHT (s)", "DFS (s)", "Status", "Speedup"]
table_data = []
cell_colors = []

family_order = {"Sparse": 0, "Medium": 1, "Dense": 2, "Tangled": 3}
sorted_recs = sorted(records,
                     key=lambda r: (family_order.get(r["family"], 9),
                                    r["genus"]))

for r in sorted_recs:
    fwht_s = (f"{r['fwht_time']:.4f}" if r["fwht_time"] < 1
              else f"{r['fwht_time']:.2f}")

    if r["dfs_status"] == "TIMEOUT":
        ext = r["dfs_extrapolated"]
        if ext >= 86400:
            dfs_s = f"\u2248{ext/86400:.1f} days"
        elif ext >= 3600:
            dfs_s = f"\u2248{ext/3600:.1f} h"
        else:
            dfs_s = f"\u2248{ext:.0f} s"
        status_s = "TIMEOUT"
        sp = ext / r["fwht_time"] if r["fwht_time"] > 0 else 0
        sp_s = f"\u2248{sp:,.0f}\u00d7"
        row_c = ["#FFCDD2"] * 9
    else:
        dfs_s = (f"{r['dfs_time']:.4f}" if r["dfs_time"] < 1
                 else f"{r['dfs_time']:.2f}")
        status_s = "OK"
        sp = r["dfs_time"] / r["fwht_time"] if r["fwht_time"] > 0 else 0
        sp_s = (f"{sp:,.1f}\u00d7" if sp < 1000
                else f"{sp:,.0f}\u00d7")
        if sp > 100:
            row_c = ["#C8E6C9"] * 9
        elif sp > 10:
            row_c = ["#FFF9C4"] * 9
        else:
            row_c = ["#FFFFFF"] * 9

    table_data.append([r["family"], r["file"].replace(".gr", ""),
                       r["N"], r["M"], r["genus"],
                       fwht_s, dfs_s, status_s, sp_s])
    cell_colors.append(row_c)

tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                cellColours=cell_colors, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.5)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#37474F")
    tbl[0, j].set_text_props(color="white", fontweight="bold", fontsize=10)

for i, r in enumerate(sorted_recs):
    if r["dfs_status"] == "TIMEOUT":
        for j in range(len(col_labels)):
            tbl[i+1, j].set_text_props(fontweight="bold")

col_w = [0.07, 0.16, 0.04, 0.04, 0.04, 0.09, 0.14, 0.07, 0.12]
for j, w in enumerate(col_w):
    for i in range(len(table_data) + 1):
        tbl[i, j].set_width(w)

ax4.set_title("RQ2: Complete Experimental Data \u2014 FWHT vs DFS\n"
              "(Red = DFS Timeout & Extrapolated, "
              "Green = Speedup > 100\u00d7)",
              fontsize=14, fontweight="bold", pad=20)

fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, "RQ2_fig4_data_table.png"),
             dpi=300, bbox_inches="tight")
print("[Saved] Fig 4")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("RQ2 Summary")
print("=" * 60)
ok_sp = [(r["dfs_time"] / r["fwht_time"], r["file"])
         for r in records if r["dfs_status"] == "OK" and r["fwht_time"] > 0]
if ok_sp:
    vals = [s for s, _ in ok_sp]
    print(f"  OK entries:      {len(ok_sp)}")
    print(f"  Median speedup:  {np.median(vals):.1f}x")
    print(f"  Max OK speedup:  {max(vals):.1f}x  "
          f"({max(ok_sp, key=lambda x: x[0])[1]})")

to_count = sum(1 for r in records if r["dfs_status"] == "TIMEOUT")
print(f"  TIMEOUT entries: {to_count}")

all_sp2 = [(r["dfs_extrapolated"] / r["fwht_time"], r["file"])
           for r in records
           if r["fwht_time"] > 0 and r["dfs_extrapolated"] > 0]
if all_sp2:
    best = max(all_sp2, key=lambda x: x[0])
    print(f"  Max speedup (incl. extrap): {best[0]:,.0f}x  ({best[1]})")

print(f"\nAll figures saved to: {FIG_DIR}")
