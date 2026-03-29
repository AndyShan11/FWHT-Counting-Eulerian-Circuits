"""
RQ1 Plotting: Execution Time and Eulerian Circuit Count vs Circuit Rank (g).

Reads pre-recorded experimental data (embedded below) and generates:
  Fig 1: Execution time vs g at fixed edge count m
  Fig 2: Eulerian circuit count vs g at fixed edge count m

Dependencies: pandas, matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FIG_DIR    = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ================================================================
# 1. Raw experimental data (from FWHT solver runs)
# ================================================================
data = """\
Filename\tEulerian Circuits\tTime Elapsed (s)
graph_m25_g10_1.gr\t8668\t0.0913628
graph_m25_g10_2.gr\t7720\t0.0945244
graph_m25_g10_3.gr\t6496\t0.0937971
graph_m25_g10_4.gr\t8328\t0.0909312
graph_m25_g10_5.gr\t7800\t0.0795801
graph_m25_g11_1.gr\t23008\t0.178949
graph_m25_g11_2.gr\t24796\t0.201482
graph_m25_g11_3.gr\t23440\t0.164532
graph_m25_g11_4.gr\t22392\t0.16051
graph_m25_g11_5.gr\t24992\t0.147467
graph_m25_g12_1.gr\t67760\t0.320508
graph_m25_g12_2.gr\t71496\t0.308774
graph_m25_g12_3.gr\t71172\t0.340047
graph_m25_g12_4.gr\t64792\t0.350412
graph_m25_g12_5.gr\t67896\t0.341924
graph_m25_g13_1.gr\t191776\t0.807879
graph_m25_g13_2.gr\t185312\t0.755394
graph_m25_g13_3.gr\t215840\t0.754275
graph_m25_g13_4.gr\t195748\t0.624497
graph_m25_g13_5.gr\t200576\t0.667577
graph_m25_g5_1.gr\t40\t0.0024361
graph_m25_g5_2.gr\t44\t0.0020558
graph_m25_g5_3.gr\t44\t0.0028993
graph_m25_g5_4.gr\t40\t0.0030235
graph_m25_g5_5.gr\t44\t0.0024498
graph_m25_g6_1.gr\t120\t0.0051797
graph_m25_g6_2.gr\t120\t0.0043363
graph_m25_g6_3.gr\t132\t0.005182
graph_m25_g6_4.gr\t120\t0.0053467
graph_m25_g6_5.gr\t120\t0.0064806
graph_m25_g7_1.gr\t304\t0.0090759
graph_m25_g7_2.gr\t304\t0.0092371
graph_m25_g7_3.gr\t352\t0.0102768
graph_m25_g7_4.gr\t304\t0.0095213
graph_m25_g7_5.gr\t304\t0.0106263
graph_m25_g8_1.gr\t1008\t0.0185739
graph_m25_g8_2.gr\t1008\t0.0183136
graph_m25_g8_3.gr\t1052\t0.0182285
graph_m25_g8_4.gr\t968\t0.0197725
graph_m25_g8_5.gr\t1052\t0.019573
graph_m25_g9_1.gr\t2808\t0.0495046
graph_m25_g9_2.gr\t2640\t0.0395941
graph_m25_g9_3.gr\t2560\t0.0389498
graph_m25_g9_4.gr\t2932\t0.0405065
graph_m25_g9_5.gr\t2592\t0.0422685
graph_m30_g10_1.gr\t39264\t0.181574
graph_m30_g10_2.gr\t11328\t0.189492
graph_m30_g10_3.gr\t6400\t0.1932
graph_m30_g11_1.gr\t40864\t0.34244
graph_m30_g11_2.gr\t17216\t0.332925
graph_m30_g11_3.gr\t54784\t0.315665
graph_m30_g12_1.gr\t172800\t0.655656
graph_m30_g12_2.gr\t50424\t0.618039
graph_m30_g12_3.gr\t408192\t0.706327
graph_m30_g13_1.gr\t1159680\t1.5672
graph_m30_g13_2.gr\t2071872\t1.42914
graph_m30_g13_3.gr\t315264\t1.42493
graph_m30_g14_1.gr\t6983680\t3.18899
graph_m30_g14_2.gr\t1716224\t3.18124
graph_m30_g14_3.gr\t6272352\t3.24639
graph_m30_g15_1.gr\t17391744\t13.3297
graph_m30_g15_2.gr\t6366432\t6.84275
graph_m30_g15_3.gr\t6987264\t19.4865
graph_m30_g16_1.gr\t554503680\t41.1097
graph_m30_g16_2.gr\t87138048\t13.9988
graph_m30_g16_3.gr\t1294341120\t14.7414
graph_m30_g17_1.gr\t644055552\t28.5273
graph_m30_g17_2.gr\t142345728\t27.214
graph_m30_g17_3.gr\t673880064\t28.9559
graph_m30_g18_1.gr\t6742711296\t58.5845
graph_m30_g18_2.gr\t21006314496\t59.3174
graph_m30_g18_3.gr\t8683905024\t59.4278
graph_m30_g5_1.gr\t48\t0.0054628
graph_m30_g5_2.gr\t36\t0.004761
graph_m30_g5_3.gr\t48\t0.0039862
graph_m30_g6_1.gr\t112\t0.0103449
graph_m30_g6_2.gr\t88\t0.0139017
graph_m30_g6_3.gr\t120\t0.0114298
graph_m30_g7_1.gr\t312\t0.0185468
graph_m30_g7_2.gr\t192\t0.0210133
graph_m30_g7_3.gr\t264\t0.0173106
graph_m30_g8_1.gr\t640\t0.0291182
graph_m30_g8_2.gr\t968\t0.0354019
graph_m30_g8_3.gr\t880\t0.0323025
graph_m30_g9_1.gr\t6656\t0.0666586
graph_m30_g9_2.gr\t4608\t0.0632051
graph_m30_g9_3.gr\t6336\t0.0720914
graph_m40_g10_1.gr\t7720\t0.289229
graph_m40_g10_2.gr\t7056\t0.306976
graph_m40_g10_3.gr\t29760\t0.312801
graph_m40_g11_1.gr\t35744\t0.62011
graph_m40_g11_2.gr\t55488\t0.606434
graph_m40_g11_3.gr\t36992\t0.594854
graph_m40_g12_1.gr\t167680\t1.46513
graph_m40_g12_2.gr\t98304\t1.3072
graph_m40_g12_3.gr\t103776\t1.25867
graph_m40_g13_1.gr\t1234176\t2.92276
graph_m40_g13_2.gr\t1829376\t2.967
graph_m40_g13_3.gr\t349952\t2.75906
graph_m40_g14_1.gr\t1585152\t6.0333
graph_m40_g14_2.gr\t5664768\t5.98444
graph_m40_g14_3.gr\t1706000\t6.01273
graph_m40_g15_1.gr\t3993472\t12.6792
graph_m40_g15_2.gr\t2627616\t12.5306
graph_m40_g15_3.gr\t11839232\t12.5686
graph_m40_g16_1.gr\t38522880\t27.8886
graph_m40_g16_2.gr\t56764416\t26.8356
graph_m40_g16_3.gr\t33480960\t26.5647
graph_m40_g17_1.gr\t65835712\t119.949
graph_m40_g17_2.gr\t84629280\t92.8685
graph_m40_g17_3.gr\t179559936\t57.854
graph_m40_g5_1.gr\t24\t0.0060592
graph_m40_g5_2.gr\t48\t0.005664
graph_m40_g5_3.gr\t44\t0.0055717
graph_m40_g6_1.gr\t48\t0.0133884
graph_m40_g6_2.gr\t120\t0.0184927
graph_m40_g6_3.gr\t88\t0.0118073
graph_m40_g7_1.gr\t264\t0.0251254
graph_m40_g7_2.gr\t352\t0.0251326
graph_m40_g7_3.gr\t512\t0.0264348
graph_m40_g8_1.gr\t800\t0.0594023
graph_m40_g8_2.gr\t896\t0.0554568
graph_m40_g8_3.gr\t832\t0.0563004
graph_m40_g9_1.gr\t1920\t0.144006
graph_m40_g9_2.gr\t3392\t0.140395
graph_m40_g9_3.gr\t3424\t0.123206
graph_m50_g10_1.gr\t14256\t0.676274
graph_m50_g10_2.gr\t4736\t0.632995
graph_m50_g10_3.gr\t3776\t0.625355
graph_m50_g11_1.gr\t18560\t1.25494
graph_m50_g11_2.gr\t15360\t1.482
graph_m50_g11_3.gr\t52096\t1.23987
graph_m50_g12_1.gr\t108224\t2.63952
graph_m50_g12_2.gr\t61552\t2.4782
graph_m50_g12_3.gr\t46800\t2.52152
graph_m50_g13_1.gr\t422400\t5.08916
graph_m50_g13_2.gr\t177408\t5.24565
graph_m50_g13_3.gr\t111392\t5.68125
graph_m50_g14_1.gr\t1754112\t11.7692
graph_m50_g14_2.gr\t2872320\t12.9891
graph_m50_g14_3.gr\t1421376\t14.7963
graph_m50_g15_1.gr\t2859520\t23.4343
graph_m50_g15_2.gr\t4094592\t23.4586
graph_m50_g15_3.gr\t5602560\t25.1193
graph_m50_g16_1.gr\t10015232\t48.5101
graph_m50_g16_2.gr\t6259456\t47.5687
graph_m50_g16_3.gr\t11644800\t61.2214
graph_m50_g17_1.gr\t15697344\t308.968
graph_m50_g17_2.gr\t37383488\t316.619
graph_m50_g17_3.gr\t35358048\t171.89
graph_m50_g5_1.gr\t32\t0.0717183
graph_m50_g5_2.gr\t16\t0.833211
graph_m50_g5_3.gr\t16\t0.639428
graph_m50_g6_1.gr\t72\t0.564659
graph_m50_g6_2.gr\t88\t0.928381
graph_m50_g6_3.gr\t80\t1.04315
graph_m50_g7_1.gr\t384\t0.940098
graph_m50_g7_2.gr\t128\t1.034
graph_m50_g7_3.gr\t328\t0.909381
graph_m50_g8_1.gr\t608\t1.00746
graph_m50_g8_2.gr\t480\t1.11071
graph_m50_g8_3.gr\t656\t1.027
graph_m50_g9_1.gr\t1792\t1.1993
graph_m50_g9_2.gr\t1920\t1.36516
graph_m50_g9_3.gr\t2848\t0.597355"""

# ================================================================
# 2. Data processing and cleaning
# ================================================================
df = pd.read_csv(io.StringIO(data), sep='\t')

# Filter out rows with ERROR status
df = df[~df['Eulerian Circuits'].astype(str).str.contains('ERROR')]

df['Eulerian Circuits'] = pd.to_numeric(df['Eulerian Circuits'])
df['Time Elapsed (s)']  = pd.to_numeric(df['Time Elapsed (s)'])

# Extract parameters m and g from filename via regex
df['m'] = df['Filename'].apply(lambda x: int(re.search(r'm(\d+)', x).group(1)))
df['g'] = df['Filename'].apply(lambda x: int(re.search(r'g(\d+)', x).group(1)))

# Aggregate by (m, g) across 3-5 repetitions; compute mean and std
agg_df = df.groupby(['m', 'g']).agg({
    'Eulerian Circuits': ['mean', 'std'],
    'Time Elapsed (s)': ['mean', 'std']
}).reset_index()

# Flatten multi-level column index
agg_df.columns = ['m', 'g', 'EC_mean', 'EC_std', 'Time_mean', 'Time_std']

# ================================================================
# 3. Publication-quality plot configuration
# ================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
})

# --------------- Fig 1: Runtime vs g ---------------
fig, ax = plt.subplots(figsize=(8, 6))
for m_val in sorted(df['m'].unique()):
    subset = agg_df[agg_df['m'] == m_val]
    ax.errorbar(subset['g'], subset['Time_mean'], yerr=subset['Time_std'],
                marker='o', label=f'm = {m_val}', capsize=5,
                linewidth=2, markersize=8)

ax.set_yscale('log')
ax.set_xlabel('Parameter $g$')
ax.set_ylabel('Execution Time (s)')
ax.legend(title='Total Edges ($m$)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'RQ1_time_vs_g.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# --------------- Fig 2: Eulerian circuit count vs g ---------------
fig, ax = plt.subplots(figsize=(8, 6))
for m_val in sorted(df['m'].unique()):
    subset = agg_df[agg_df['m'] == m_val]
    ax.errorbar(subset['g'], subset['EC_mean'], yerr=subset['EC_std'],
                marker='s', label=f'm = {m_val}', capsize=5,
                linewidth=2, markersize=8)

ax.set_yscale('log')
ax.set_xlabel('Parameter $g$')
ax.set_ylabel('Number of Eulerian Circuits')
ax.legend(title='Total Edges ($m$)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'RQ1_ec_vs_g.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print("Saved: RQ1_time_vs_g.png, RQ1_ec_vs_g.png")
