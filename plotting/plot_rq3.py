"""
RQ3 Plotting: Effect of Graph Size (M) at Fixed Circuit Rank (g).

Reads the CSV produced by fwht_solver and generates publication-quality figures:
  Fig 1: T vs M for g=8,10,12 with O(m^3 log m) fit, error bars, 3 families
  Fig 2: T / (m^3 log m) normalized — should be flat if complexity is correct
  Fig 3: Eulerian circuit count vs M at each g

Dependencies: numpy, matplotlib, scipy
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FIG_DIR    = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 0. Load data
# ============================================================
CSV_FILE = os.path.join(REPO_ROOT, "results", "eulerian_RQ3_results.csv")

# Parse filename pattern: {family}_g{g}_M{M}_r{rep}.gr
pattern = re.compile(r'^(\w+)_g(\d+)_M(\d+)_r(\d+)\.gr$')
partial_pattern = re.compile(r'^PARTIAL\((\d+)/(\d+)\)$')

# data[g][(family, M)] = list of (time, ec, is_partial)
data = defaultdict(lambda: defaultdict(list))

with open(CSV_FILE, 'r') as f:
    header = f.readline()
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        fname = parts[0]
        match = pattern.match(fname)
        if not match:
            continue
        family = match.group(1)
        g = int(match.group(2))
        M = int(match.group(3))
        ec_str = parts[1].strip()
        time_str = parts[2].strip()
        status = parts[3].strip() if len(parts) >= 4 else 'COMPLETE'

        if ec_str in ('ERROR',) or time_str == 'N/A':
            continue
        try:
            ec = int(ec_str)
            t = float(time_str)
        except ValueError:
            continue

        is_partial = False
        pm = partial_pattern.match(status)
        if pm:
            is_partial = True
            completed = int(pm.group(1))
            total = int(pm.group(2))
            if completed > 0 and completed < total:
                ec = int(ec * total / completed)

        data[g][(family, M)].append((t, ec, is_partial))

# ============================================================
# Style setup
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
})

COLORS_G = {8: '#1f77b4', 10: '#d62728', 12: '#2ca02c'}
MARKERS_FAM = {'cactus': 's', 'chain': '^', 'random': 'o'}
LABELS_FAM = {'cactus': 'Cactus', 'chain': 'Chain-of-Cycles',
              'random': 'Random Eulerian'}

# ============================================================
# 1. Main figure: T vs M with polynomial fit
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300, sharey=False)

for idx, g in enumerate(sorted(data.keys())):
    ax = axes[idx]
    gdata = data[g]

    fam_complete = defaultdict(lambda: defaultdict(list))
    fam_partial  = defaultdict(lambda: defaultdict(list))
    for (fam, M), vals in gdata.items():
        for t, ec, is_partial in vals:
            if is_partial:
                fam_partial[fam][M].append(t)
            else:
                fam_complete[fam][M].append(t)

    for fam in ['cactus', 'chain', 'random']:
        color = None
        if fam in fam_complete and fam_complete[fam]:
            Ms = sorted(fam_complete[fam].keys())
            means = [np.mean(fam_complete[fam][m]) for m in Ms]
            stds  = [np.std(fam_complete[fam][m]) for m in Ms]
            line = ax.errorbar(Ms, means, yerr=stds,
                               marker=MARKERS_FAM.get(fam, 'o'),
                               markersize=6, capsize=3, linewidth=1.5,
                               label=LABELS_FAM.get(fam, fam), alpha=0.85)
            color = line[0].get_color()

        if fam in fam_partial and fam_partial[fam]:
            Ms_p = sorted(fam_partial[fam].keys())
            means_p = [np.mean(fam_partial[fam][m]) for m in Ms_p]
            stds_p  = [np.std(fam_partial[fam][m]) for m in Ms_p]
            lbl = (LABELS_FAM.get(fam, fam) + ' (predicted)'
                   if color is None else None)
            ax.errorbar(Ms_p, means_p, yerr=stds_p,
                        marker=MARKERS_FAM.get(fam, 'o'),
                        markersize=6, capsize=3, linewidth=1.5,
                        linestyle='--', markerfacecolor='none',
                        color=color, label=lbl, alpha=0.6)

    # Fit T = C * m^3 * log(m) to complete data
    all_M, all_T = [], []
    for (fam, M), vals in gdata.items():
        for t, ec, is_partial in vals:
            if not is_partial:
                all_M.append(M)
                all_T.append(t)
    all_M = np.array(all_M, dtype=float)
    all_T = np.array(all_T, dtype=float)

    if len(all_M) > 2:
        def model(m, C):
            return C * m**3 * np.log2(m)
        try:
            popt, _ = curve_fit(model, all_M, all_T, p0=[1e-8])
            m_fit = np.linspace(all_M.min(), all_M.max(), 200)
            t_fit = model(m_fit, *popt)
            ax.plot(m_fit, t_fit, 'k--', linewidth=1.2, alpha=0.6,
                    label=r'Fit: $C \cdot m^3 \log m$')
        except Exception:
            pass

    ax.set_xlabel('Number of Edges ($m$)', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'$g = {g}$', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=10)

fig.suptitle('RQ3: Execution Time vs. Graph Size at Fixed Circuit Rank',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'RQ3_time_vs_M.png'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'RQ3_time_vs_M.pdf'),
            bbox_inches='tight')
print("[+] Saved RQ3_time_vs_M.png / .pdf")

# ============================================================
# 2. Normalized plot: T / (m^3 log m)
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=True)

for idx, g in enumerate(sorted(data.keys())):
    ax = axes2[idx]
    gdata = data[g]

    for fam in ['cactus', 'chain', 'random']:
        fam_items = {M: vals for (f, M), vals in gdata.items() if f == fam}
        if not fam_items:
            continue

        complete_M = defaultdict(list)
        partial_M  = defaultdict(list)
        for M, vals in fam_items.items():
            for t, ec, is_partial in vals:
                if is_partial:
                    partial_M[M].append(t)
                else:
                    complete_M[M].append(t)

        color = None
        if complete_M:
            Ms = sorted(complete_M.keys())
            norm_means, norm_stds = [], []
            for m_val in Ms:
                denom = m_val**3 * np.log2(m_val)
                normed = [t / denom for t in complete_M[m_val]]
                norm_means.append(np.mean(normed))
                norm_stds.append(np.std(normed))
            line = ax.errorbar(Ms, norm_means, yerr=norm_stds,
                               marker=MARKERS_FAM.get(fam, 'o'),
                               markersize=6, capsize=3, linewidth=1.5,
                               label=LABELS_FAM.get(fam, fam), alpha=0.85)
            color = line[0].get_color()

        if partial_M:
            Ms_p = sorted(partial_M.keys())
            norm_means_p, norm_stds_p = [], []
            for m_val in Ms_p:
                denom = m_val**3 * np.log2(m_val)
                normed = [t / denom for t in partial_M[m_val]]
                norm_means_p.append(np.mean(normed))
                norm_stds_p.append(np.std(normed))
            ax.errorbar(Ms_p, norm_means_p, yerr=norm_stds_p,
                        marker=MARKERS_FAM.get(fam, 'o'),
                        markersize=6, capsize=3, linewidth=1.5,
                        linestyle='--', markerfacecolor='none',
                        color=color, alpha=0.6)

    ax.set_xlabel('Number of Edges ($m$)', fontsize=12)
    if idx == 0:
        ax.set_ylabel(r'$T \;/\; (m^3 \log m)$', fontsize=12)
    ax.set_title(f'$g = {g}$', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))

fig2.suptitle('RQ3: Normalized Time (Complexity Validation)',
              fontsize=15, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'RQ3_normalized.png'),
             dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(FIG_DIR, 'RQ3_normalized.pdf'),
             bbox_inches='tight')
print("[+] Saved RQ3_normalized.png / .pdf")

# ============================================================
# 3. Eulerian circuit count vs M
# ============================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

for idx, g in enumerate(sorted(data.keys())):
    ax = axes3[idx]
    gdata = data[g]

    for fam in ['cactus', 'chain', 'random']:
        fam_items = {M: vals for (f, M), vals in gdata.items() if f == fam}
        if not fam_items:
            continue

        complete_M = defaultdict(list)
        partial_M  = defaultdict(list)
        for M, vals in fam_items.items():
            for t, ec, is_partial in vals:
                if is_partial:
                    partial_M[M].append(ec)
                else:
                    complete_M[M].append(ec)

        color = None
        if complete_M:
            Ms = sorted(complete_M.keys())
            ec_means = [np.mean(complete_M[m]) for m in Ms]
            ec_stds  = [np.std(complete_M[m]) for m in Ms]
            line = ax.errorbar(Ms, ec_means, yerr=ec_stds,
                               marker=MARKERS_FAM.get(fam, 'o'),
                               markersize=6, capsize=3, linewidth=1.5,
                               label=LABELS_FAM.get(fam, fam), alpha=0.85)
            color = line[0].get_color()

        if partial_M:
            Ms_p = sorted(partial_M.keys())
            ec_means_p = [np.mean(partial_M[m]) for m in Ms_p]
            ec_stds_p  = [np.std(partial_M[m]) for m in Ms_p]
            lbl = (LABELS_FAM.get(fam, fam) + ' (predicted)'
                   if color is None else None)
            ax.errorbar(Ms_p, ec_means_p, yerr=ec_stds_p,
                        marker=MARKERS_FAM.get(fam, 'o'),
                        markersize=6, capsize=3, linewidth=1.5,
                        linestyle='--', markerfacecolor='none',
                        color=color, label=lbl, alpha=0.6)

    ax.set_xlabel('Number of Edges ($m$)', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Eulerian Circuit Count', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'$g = {g}$', fontsize=13, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    dash_handle = Line2D([0], [0], color='gray', linestyle='--',
                         marker='o', markerfacecolor='none', markersize=5,
                         label='Predicted (partial)')
    handles.append(dash_handle)
    labels.append('Predicted (partial)')
    ax.legend(handles=handles, labels=labels, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

fig3.suptitle('RQ3: Eulerian Circuit Count vs. Graph Size',
              fontsize=15, fontweight='bold', y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'RQ3_ec_count.png'),
             dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(FIG_DIR, 'RQ3_ec_count.pdf'),
             bbox_inches='tight')
print("[+] Saved RQ3_ec_count.png / .pdf")

print("\n[+] All RQ3 plots generated successfully.")
