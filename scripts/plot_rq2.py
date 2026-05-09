import matplotlib.pyplot as plt
import numpy as np

# Data from user's experiment
g_values = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

# FWHT Time Elapsed (s)
# Using 600 for Timeout at g=15
fwht_time = np.array([0.0076333, 0.0054338, 0.0143224, 0.0587973, 0.721703, 2.05059, 1.83523, 23.0069, 140.165, 600.0])

# DFS Time Elapsed (s)
# Using 600 for Timeouts (g=11, 13, 14, 15)
dfs_time = np.array([0.0016167, 0.0012954, 0.0086417, 0.25809, 19.5756, 600.0, 209.435, 600.0, 600.0, 600.0])

# Configure plotting style suitable for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(9, 6), dpi=300)

# Plot DFS (red, dashed, squares)
plt.plot(g_values, dfs_time, marker='s', linestyle='--', linewidth=2.5, markersize=8, color='#d62728', label='DFS (Pure Combinatorial)')

# Plot FWHT (blue, solid, circles)
plt.plot(g_values, fwht_time, marker='o', linestyle='-', linewidth=2.5, markersize=8, color='#1f77b4', label='FWHT (Algebraic Graph Theory)')

# Y-axis log scale
plt.yscale('log')

# Add a horizontal line for Timeout
plt.axhline(y=600, color='#7f7f7f', linestyle=':', linewidth=2, label='Timeout (600s)')

# Annotate the Crossover point
# Crossover is between g=8 and g=9 (where DFS jumps above FWHT)
plt.annotate('Phase Transition\n(Crossover Point)', 
             xy=(8.5, 0.05), xytext=(9.5, 0.002),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold', ha='center')

# Labels and title
plt.xlabel(r'Graph Complexity / Genus ($g$)', fontsize=14, fontweight='bold')
plt.ylabel('Time Elapsed (seconds, Log Scale)', fontsize=14, fontweight='bold')
plt.title('Performance Crossover: FWHT vs DFS', fontsize=16, fontweight='bold', pad=15)

# Formatting ticks
plt.xticks(g_values, fontsize=12)
plt.yticks(fontsize=12)

# Legend
plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('crossover_plot.png')