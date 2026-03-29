"""
RQ4 Plotting: Parallel Scaling Analysis.

Reads rq4_results.csv and produces publication-quality figures:
  Fig 1: Speedup vs Threads for multiple problem sizes (with ideal reference)
  Fig 2: Parallel Efficiency (speedup / threads) vs Threads
  Fig 3: Amdahl's Law fit — estimate serial fraction f

Dependencies: numpy, matplotlib
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FIG_DIR    = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 0. Load data
# ============================================================
CSV_FILE = os.path.join(REPO_ROOT, "results", "rq4_results.csv")

# data[graph_name][threads] = list of times
data = defaultdict(lambda: defaultdict(list))

# Parse graph name pattern: rq4_g{g}_M{M}.gr
g_pattern = re.compile(r'rq4_g(\d+)_M(\d+)\.gr')

with open(CSV_FILE, 'r') as f:
    header = f.readline()
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        gname = parts[0].strip()
        try:
            threads = int(parts[1].strip())
            t = float(parts[3].strip())
        except (ValueError, IndexError):
            continue
        data[gname][threads].append(t)

# Sort graphs by g value
graph_names = sorted(data.keys(), key=lambda x: (
    int(g_pattern.match(x).group(1)) if g_pattern.match(x) else 0
))

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
})

COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
MARKERS = ['o', 's', '^', 'D', 'v']

# ============================================================
# Compute speedup and efficiency
# ============================================================
all_threads = sorted(set(t for gd in data.values() for t in gd.keys()))

graph_stats = {}
for gname in graph_names:
    gd = data[gname]
    if 1 not in gd:
        continue
    t1_mean = np.mean(gd[1])

    stats = {}
    for thr in all_threads:
        if thr not in gd:
            continue
        times = np.array(gd[thr])
        speedups = t1_mean / times
        effs = speedups / thr
        stats[thr] = {
            'speedup_mean': np.mean(speedups),
            'speedup_std': np.std(speedups),
            'eff_mean': np.mean(effs),
            'eff_std': np.std(effs),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
        }
    graph_stats[gname] = stats

valid_graphs = [g for g in graph_names if g in graph_stats]


def get_label(gname):
    m = g_pattern.match(gname)
    if m:
        return f"$g={m.group(1)},\\; m={m.group(2)}$"
    return gname


# ============================================================
# Fig 1: Speedup vs Threads (bar chart + ideal line)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(11, 6.5), dpi=300)

n_graphs = len(graph_stats)
bar_width = 0.8 / max(n_graphs, 1)
x_pos = np.arange(len(all_threads))

for gi, gname in enumerate(valid_graphs):
    stats = graph_stats[gname]
    thrs = sorted(stats.keys())
    means = [stats[t]['speedup_mean'] for t in thrs]
    stds  = [stats[t]['speedup_std'] for t in thrs]
    offsets = (x_pos[:len(thrs)]
               + gi * bar_width - (n_graphs - 1) * bar_width / 2)

    bars = ax1.bar(offsets, means, bar_width * 0.9, yerr=stds,
                   capsize=2, color=COLORS[gi % len(COLORS)], alpha=0.8,
                   edgecolor='black', linewidth=0.5,
                   label=get_label(gname))

    # Annotate speedup and time on bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        t_mean = stats[thrs[i]]['time_mean']
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.15,
                 f'{yval:.1f}x', ha='center', va='bottom',
                 fontsize=7, fontweight='bold')
        if t_mean >= 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2, yval * 0.4,
                     f'{t_mean:.2f}s', ha='center', va='center',
                     fontsize=6, color='white', fontweight='bold')

# Ideal linear scaling line
ideal = [float(t) for t in all_threads]
ax1.plot(x_pos[:len(all_threads)], ideal[:len(all_threads)],
         'k--', linewidth=2, alpha=0.5, marker='',
         label='Ideal Linear Scaling')

ax1.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax1.set_ylabel('Speedup Ratio', fontsize=13, fontweight='bold')
ax1.set_title('RQ4: Multi-threading Speedup across Problem Sizes',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(all_threads, fontsize=11)
ax1.tick_params(labelsize=11)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(bottom=0)

fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'RQ4_speedup.png'),
             dpi=300, bbox_inches='tight')
fig1.savefig(os.path.join(FIG_DIR, 'RQ4_speedup.pdf'),
             bbox_inches='tight')
print("[+] Saved RQ4_speedup.png / .pdf")

# ============================================================
# Fig 2: Parallel Efficiency
# ============================================================
fig2, ax2 = plt.subplots(figsize=(9, 6), dpi=300)

for gi, gname in enumerate(valid_graphs):
    stats = graph_stats[gname]
    thrs = sorted(stats.keys())
    effs = [stats[t]['eff_mean'] for t in thrs]
    estd = [stats[t]['eff_std'] for t in thrs]

    ax2.errorbar(thrs, effs, yerr=estd,
                 marker=MARKERS[gi % len(MARKERS)], markersize=8,
                 color=COLORS[gi % len(COLORS)],
                 linewidth=2, capsize=4, label=get_label(gname))

ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
            alpha=0.6, label='Ideal (100%)')

# Auto-scale y-axis
all_eff_vals = []
for gname in valid_graphs:
    for t in graph_stats[gname]:
        s = graph_stats[gname][t]
        all_eff_vals.append(s['eff_mean'] + s['eff_std'])
        all_eff_vals.append(s['eff_mean'] - s['eff_std'])
y_max = max(all_eff_vals) * 1.1
y_max = max(y_max, 1.2)
ax2.set_ylim(0, y_max)

# Shade superlinear region
ax2.fill_between([0.8, max(all_threads) * 1.3], 1.0, y_max,
                 alpha=0.06, color='#ff8800', zorder=0)
ax2.text(1.2, y_max * 0.92, 'Superlinear region\n(cache / memory effects)',
         fontsize=8.5, color='#996600', style='italic', va='top')

ax2.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax2.set_ylabel('Parallel Efficiency (Speedup / Threads)', fontsize=13,
               fontweight='bold')
ax2.set_title('RQ4: Parallel Efficiency vs. Thread Count',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xscale('log', base=2)
ax2.set_xticks(all_threads)
ax2.set_xticklabels(all_threads, fontsize=11)
ax2.set_xlim(0.8, max(all_threads) * 1.3)
ax2.tick_params(labelsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.4)

fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'RQ4_efficiency.png'),
             dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(FIG_DIR, 'RQ4_efficiency.pdf'),
             bbox_inches='tight')
print("[+] Saved RQ4_efficiency.png / .pdf")

# ============================================================
# Fig 3: Amdahl's Law Fit
# ============================================================
fig3, ax3 = plt.subplots(figsize=(9, 5.5), dpi=300)


def amdahl(p, f):
    """Amdahl's law: S(p) = 1 / (f + (1-f)/p)"""
    return 1.0 / (f + (1.0 - f) / p)


def fit_amdahl_simple(thrs, speedups):
    """Least-squares fit for serial fraction f using grid search."""
    best_f, best_err = 0.5, float('inf')
    for f_candidate in np.linspace(0.001, 0.999, 2000):
        predicted = np.array([amdahl(p, f_candidate) for p in thrs])
        err = np.sum((predicted - speedups) ** 2)
        if err < best_err:
            best_err = err
            best_f = f_candidate
    return best_f


for gi, gname in enumerate(valid_graphs):
    stats = graph_stats[gname]
    thrs = sorted(stats.keys())
    sp_vals = np.array([stats[t]['speedup_mean'] for t in thrs])
    thr_arr = np.array(thrs, dtype=float)

    ax3.plot(thr_arr, sp_vals,
             marker=MARKERS[gi % len(MARKERS)], markersize=7,
             color=COLORS[gi % len(COLORS)],
             linewidth=1.8, label=get_label(gname))

    # Fit Amdahl's law
    try:
        f_serial = fit_amdahl_simple(thr_arr, sp_vals)
        p_fit = np.linspace(1, max(all_threads), 200)
        s_fit = np.array([amdahl(p, f_serial) for p in p_fit])
        ax3.plot(p_fit, s_fit, '--', color=COLORS[gi % len(COLORS)],
                 alpha=0.5, linewidth=1.2)
        m_match = g_pattern.match(gname)
        g_val = m_match.group(1) if m_match else '?'
        ax3.annotate(
            f'$f={f_serial:.3f}$\n($g={g_val}$)',
            xy=(max(all_threads) * 0.85,
                amdahl(max(all_threads), f_serial)),
            fontsize=8, color=COLORS[gi % len(COLORS)], ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.7))
    except Exception:
        pass

# Ideal scaling line
p_ideal = np.linspace(1, max(all_threads), 200)
ax3.plot(p_ideal, p_ideal, 'k--', linewidth=1.5, alpha=0.4,
         label='Ideal ($f=0$)')

ax3.set_xlabel('Number of Threads ($p$)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Speedup $S(p)$', fontsize=13, fontweight='bold')
ax3.set_title("RQ4: Amdahl's Law Analysis",
              fontsize=14, fontweight='bold', pad=15)
ax3.set_xscale('log', base=2)
ax3.set_xticks(all_threads)
ax3.set_xticklabels(all_threads, fontsize=11)
ax3.tick_params(labelsize=11)
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, linestyle='--', alpha=0.4)

fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'RQ4_amdahl.png'),
             dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(FIG_DIR, 'RQ4_amdahl.pdf'),
             bbox_inches='tight')
print("[+] Saved RQ4_amdahl.png / .pdf")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 80)
print("RQ4 SUMMARY TABLE")
print("=" * 80)
print(f"{'Graph':<25} {'Threads':<10} {'Time(s)':<12} "
      f"{'Speedup':<10} {'Efficiency':<12}")
print("-" * 80)
for gname in valid_graphs:
    stats = graph_stats[gname]
    for thr in sorted(stats.keys()):
        s = stats[thr]
        print(f"{gname:<25} {thr:<10} {s['time_mean']:<12.4f} "
              f"{s['speedup_mean']:<10.2f} {s['eff_mean']:<12.3f}")

print("\n[+] All RQ4 plots generated successfully.")
