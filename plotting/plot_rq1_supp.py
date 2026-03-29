"""
RQ1 Supplementary Figure: Eulerian Circuits vs Edge Count at Fixed Circuit Rank.

Shows how ec(G) varies with m at fixed g = {10, 12, 14}, with individual
data points (scatter) and average trend lines.

Dependencies: matplotlib, numpy
"""

import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FIG_DIR    = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============ Pre-recorded data: (g, m) -> [ec1, ec2, ec3] ============
data = {
    # g=10
    (10, 18): [13920, 22464, 53280],
    (10, 22): [22144, 23232, 21184],
    (10, 27): [13840, 13488, 32736],
    (10, 35): [8768, 7184, 14720],
    (10, 50): [4896, 9024, 9216],
    # g=12
    (12, 22): [1683072, 1250304, 1206144],
    (12, 27): [300544, 500992, 327616],
    (12, 33): [325120, 133696, 102288],
    (12, 40): [446016, 43840, 167808],
    (12, 50): [102912, 102080, 113920],
    # g=14
    (14, 26): [6230784, 4384896, 10382400],
    (14, 32): [8908800, 7580160, 6066432],
    (14, 40): [1332224, 1673728, 1223296],
    (14, 50): [754304, 544896, 1354752],
}

g_colors  = {10: '#2196F3', 12: '#FF9800', 14: '#4CAF50'}
g_markers = {10: 'o', 12: 's', 14: '^'}

fig, ax = plt.subplots(figsize=(8, 5.5))

for g in [10, 12, 14]:
    ms = sorted([m for (gg, m) in data if gg == g])
    avg_ec = [np.mean(data[(g, m)]) for m in ms]
    # Individual data points as scatter
    for m in ms:
        for ec_val in data[(g, m)]:
            ax.scatter(m, ec_val, color=g_colors[g], alpha=0.25, s=20,
                       zorder=2, edgecolors='none')
    # Average trend line
    ax.plot(ms, avg_ec, marker=g_markers[g], color=g_colors[g],
            linewidth=2, markersize=7, label=f'g = {g}', zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Number of Edges (m)', fontsize=13)
ax.set_ylabel('Number of Eulerian Circuits', fontsize=13)
ax.set_title('Eulerian Circuits vs Edge Count at Fixed Circuit Rank',
             fontsize=14)

ax.set_xticks([20, 25, 30, 35, 40, 45, 50])
ax.grid(True, which='major', linestyle='-', alpha=0.3)
ax.grid(True, which='minor', linestyle=':', alpha=0.15)
ax.legend(fontsize=12, loc='upper right')

# Secondary x-axis: approximate average degree (using g=12 as reference)
ax2 = ax.twiny()
ref_m = [20, 30, 40, 50]
ref_deg = [2 * m / (m - 12 + 1) for m in ref_m]
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ref_m)
ax2.set_xticklabels([f'{d:.1f}' for d in ref_deg], fontsize=9, color='#888')
ax2.set_xlabel('Approx. Avg Degree (g=12 ref)', fontsize=10, color='#888')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'RQ1_ec_vs_edges.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved: RQ1_ec_vs_edges.png")
