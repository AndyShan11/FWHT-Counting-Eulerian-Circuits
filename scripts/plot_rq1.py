import pandas as pd
import matplotlib.pyplot as plt
import io
import re

# ================================
# 1. 填入你的原始实验数据
# ================================
data = """Filename	Eulerian Circuits	Time Elapsed (s)
graph_m25_g10_1.gr	8668	0.0913628
graph_m25_g10_2.gr	7720	0.0945244
graph_m25_g10_3.gr	6496	0.0937971
graph_m25_g10_4.gr	8328	0.0909312
graph_m25_g10_5.gr	7800	0.0795801
graph_m25_g11_1.gr	23008	0.178949
graph_m25_g11_2.gr	24796	0.201482
graph_m25_g11_3.gr	23440	0.164532
graph_m25_g11_4.gr	22392	0.16051
graph_m25_g11_5.gr	24992	0.147467
graph_m25_g12_1.gr	67760	0.320508
graph_m25_g12_2.gr	71496	0.308774
graph_m25_g12_3.gr	71172	0.340047
graph_m25_g12_4.gr	64792	0.350412
graph_m25_g12_5.gr	67896	0.341924
graph_m25_g13_1.gr	191776	0.807879
graph_m25_g13_2.gr	185312	0.755394
graph_m25_g13_3.gr	215840	0.754275
graph_m25_g13_4.gr	195748	0.624497
graph_m25_g13_5.gr	200576	0.667577
graph_m25_g5_1.gr	40	0.0024361
graph_m25_g5_2.gr	44	0.0020558
graph_m25_g5_3.gr	44	0.0028993
graph_m25_g5_4.gr	40	0.0030235
graph_m25_g5_5.gr	44	0.0024498
graph_m25_g6_1.gr	120	0.0051797
graph_m25_g6_2.gr	120	0.0043363
graph_m25_g6_3.gr	132	0.005182
graph_m25_g6_4.gr	120	0.0053467
graph_m25_g6_5.gr	120	0.0064806
graph_m25_g7_1.gr	304	0.0090759
graph_m25_g7_2.gr	304	0.0092371
graph_m25_g7_3.gr	352	0.0102768
graph_m25_g7_4.gr	304	0.0095213
graph_m25_g7_5.gr	304	0.0106263
graph_m25_g8_1.gr	1008	0.0185739
graph_m25_g8_2.gr	1008	0.0183136
graph_m25_g8_3.gr	1052	0.0182285
graph_m25_g8_4.gr	968	0.0197725
graph_m25_g8_5.gr	1052	0.019573
graph_m25_g9_1.gr	2808	0.0495046
graph_m25_g9_2.gr	2640	0.0395941
graph_m25_g9_3.gr	2560	0.0389498
graph_m25_g9_4.gr	2932	0.0405065
graph_m25_g9_5.gr	2592	0.0422685
graph_m30_g10_1.gr	39264	0.181574
graph_m30_g10_2.gr	11328	0.189492
graph_m30_g10_3.gr	6400	0.1932
graph_m30_g11_1.gr	40864	0.34244
graph_m30_g11_2.gr	17216	0.332925
graph_m30_g11_3.gr	54784	0.315665
graph_m30_g12_1.gr	172800	0.655656
graph_m30_g12_2.gr	50424	0.618039
graph_m30_g12_3.gr	408192	0.706327
graph_m30_g13_1.gr	1159680	1.5672
graph_m30_g13_2.gr	2071872	1.42914
graph_m30_g13_3.gr	315264	1.42493
graph_m30_g14_1.gr	6983680	3.18899
graph_m30_g14_2.gr	1716224	3.18124
graph_m30_g14_3.gr	6272352	3.24639
graph_m30_g15_1.gr	17391744	13.3297
graph_m30_g15_2.gr	6366432	6.84275
graph_m30_g15_3.gr	6987264	19.4865
graph_m30_g16_1.gr	554503680	41.1097
graph_m30_g16_2.gr	87138048	13.9988
graph_m30_g16_3.gr	1294341120	14.7414
graph_m30_g17_1.gr	644055552	28.5273
graph_m30_g17_2.gr	142345728	27.214
graph_m30_g17_3.gr	673880064	28.9559
graph_m30_g18_1.gr	6742711296	58.5845
graph_m30_g18_2.gr	21006314496	59.3174
graph_m30_g18_3.gr	8683905024	59.4278
graph_m30_g5_1.gr	48	0.0054628
graph_m30_g5_2.gr	36	0.004761
graph_m30_g5_3.gr	48	0.0039862
graph_m30_g6_1.gr	112	0.0103449
graph_m30_g6_2.gr	88	0.0139017
graph_m30_g6_3.gr	120	0.0114298
graph_m30_g7_1.gr	312	0.0185468
graph_m30_g7_2.gr	192	0.0210133
graph_m30_g7_3.gr	264	0.0173106
graph_m30_g8_1.gr	640	0.0291182
graph_m30_g8_2.gr	968	0.0354019
graph_m30_g8_3.gr	880	0.0323025
graph_m30_g9_1.gr	6656	0.0666586
graph_m30_g9_2.gr	4608	0.0632051
graph_m30_g9_3.gr	6336	0.0720914
graph_m40_g10_1.gr	7720	0.289229
graph_m40_g10_2.gr	7056	0.306976
graph_m40_g10_3.gr	29760	0.312801
graph_m40_g11_1.gr	35744	0.62011
graph_m40_g11_2.gr	55488	0.606434
graph_m40_g11_3.gr	36992	0.594854
graph_m40_g12_1.gr	167680	1.46513
graph_m40_g12_2.gr	98304	1.3072
graph_m40_g12_3.gr	103776	1.25867
graph_m40_g13_1.gr	1234176	2.92276
graph_m40_g13_2.gr	1829376	2.967
graph_m40_g13_3.gr	349952	2.75906
graph_m40_g14_1.gr	1585152	6.0333
graph_m40_g14_2.gr	5664768	5.98444
graph_m40_g14_3.gr	1706000	6.01273
graph_m40_g15_1.gr	3993472	12.6792
graph_m40_g15_2.gr	2627616	12.5306
graph_m40_g15_3.gr	11839232	12.5686
graph_m40_g16_1.gr	38522880	27.8886
graph_m40_g16_2.gr	56764416	26.8356
graph_m40_g16_3.gr	33480960	26.5647
graph_m40_g17_1.gr	65835712	119.949
graph_m40_g17_2.gr	84629280	92.8685
graph_m40_g17_3.gr	179559936	57.854
graph_m40_g18_1.gr	ERROR: Cannot open graph file.	N/A
graph_m40_g18_2.gr	ERROR: Cannot open graph file.	N/A
graph_m40_g18_3.gr	ERROR: Cannot open graph file.	N/A
graph_m40_g5_1.gr	24	0.0060592
graph_m40_g5_2.gr	48	0.005664
graph_m40_g5_3.gr	44	0.0055717
graph_m40_g6_1.gr	48	0.0133884
graph_m40_g6_2.gr	120	0.0184927
graph_m40_g6_3.gr	88	0.0118073
graph_m40_g7_1.gr	264	0.0251254
graph_m40_g7_2.gr	352	0.0251326
graph_m40_g7_3.gr	512	0.0264348
graph_m40_g8_1.gr	800	0.0594023
graph_m40_g8_2.gr	896	0.0554568
graph_m40_g8_3.gr	832	0.0563004
graph_m40_g9_1.gr	1920	0.144006
graph_m40_g9_2.gr	3392	0.140395
graph_m40_g9_3.gr	3424	0.123206
graph_m50_g10_1.gr	14256	0.676274
graph_m50_g10_2.gr	4736	0.632995
graph_m50_g10_3.gr	3776	0.625355
graph_m50_g11_1.gr	18560	1.25494
graph_m50_g11_2.gr	15360	1.482
graph_m50_g11_3.gr	52096	1.23987
graph_m50_g12_1.gr	108224	2.63952
graph_m50_g12_2.gr	61552	2.4782
graph_m50_g12_3.gr	46800	2.52152
graph_m50_g13_1.gr	422400	5.08916
graph_m50_g13_2.gr	177408	5.24565
graph_m50_g13_3.gr	111392	5.68125
graph_m50_g14_1.gr	1754112	11.7692
graph_m50_g14_2.gr	2872320	12.9891
graph_m50_g14_3.gr	1421376	14.7963
graph_m50_g15_1.gr	2859520	23.4343
graph_m50_g15_2.gr	4094592	23.4586
graph_m50_g15_3.gr	5602560	25.1193
graph_m50_g16_1.gr	10015232	48.5101
graph_m50_g16_2.gr	6259456	47.5687
graph_m50_g16_3.gr	11644800	61.2214
graph_m50_g17_1.gr	15697344	308.968
graph_m50_g17_2.gr	37383488	316.619
graph_m50_g17_3.gr	35358048	171.89
graph_m50_g5_1.gr	32	0.0717183
graph_m50_g5_2.gr	16	0.833211
graph_m50_g5_3.gr	16	0.639428
graph_m50_g6_1.gr	72	0.564659
graph_m50_g6_2.gr	88	0.928381
graph_m50_g6_3.gr	80	1.04315
graph_m50_g7_1.gr	384	0.940098
graph_m50_g7_2.gr	128	1.034
graph_m50_g7_3.gr	328	0.909381
graph_m50_g8_1.gr	608	1.00746
graph_m50_g8_2.gr	480	1.11071
graph_m50_g8_3.gr	656	1.027
graph_m50_g9_1.gr	1792	1.1993
graph_m50_g9_2.gr	1920	1.36516
graph_m50_g9_3.gr	2848	0.597355"""

# ================================
# 2. 数据处理与清洗
# ================================
# 将文本转化为 DataFrame
df = pd.read_csv(io.StringIO(data), sep='\t')

# 过滤因图太密集导致出错 (如 ERROR: Cannot open graph file.) 的数据行
df = df[~df['Eulerian Circuits'].astype(str).str.contains('ERROR')]

# 转换格式为数值类型
df['Eulerian Circuits'] = pd.to_numeric(df['Eulerian Circuits'])
df['Time Elapsed (s)'] = pd.to_numeric(df['Time Elapsed (s)'])

# 通过正则表达式从文件名中提取实验参数 m 和 g
df['m'] = df['Filename'].apply(lambda x: int(re.search(r'm(\d+)', x).group(1)))
df['g'] = df['Filename'].apply(lambda x: int(re.search(r'g(\d+)', x).group(1)))

# 由于每组 m,g 会跑3~5次，这里按 m 和 g 进行聚合求均值和标准差，准备画误差棒
agg_df = df.groupby(['m', 'g']).agg({
    'Eulerian Circuits': ['mean', 'std'],
    'Time Elapsed (s)': ['mean', 'std']
}).reset_index()

# 展平多层索引，重命名列
agg_df.columns = ['m', 'g', 'EC_mean', 'EC_std', 'Time_mean', 'Time_std']

# ================================
# 3. 论文级高质量绘图配置
# ================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14, 
    'font.family': 'serif', # 衬线体更适合论文
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
})

# --------------- 图1：运行时间 vs g -------------------
fig, ax = plt.subplots(figsize=(8, 6))
# 取出实验里的所有 m 并循环作图（25, 30, 40, 50）
for m_val in sorted(df['m'].unique()):
    subset = agg_df[agg_df['m'] == m_val]
    ax.errorbar(subset['g'], subset['Time_mean'], yerr=subset['Time_std'], 
                marker='o', label=f'm = {m_val}', capsize=5, linewidth=2, markersize=8)

ax.set_yscale('log') # 呈指数级增长，采用对数坐标系
ax.set_xlabel('Parameter $g$')
ax.set_ylabel('Execution Time (s)')
ax.legend(title='Total Edges ($m$)')
plt.tight_layout()
plt.savefig('time_vs_g.png', dpi=300, bbox_inches='tight') # dpi=300高清打印标准
plt.close()


# --------------- 图2：欧拉回路数量 vs g -------------------
fig, ax = plt.subplots(figsize=(8, 6))
# 换一个marker ('s' 方块) 区分一下
for m_val in sorted(df['m'].unique()):
    subset = agg_df[agg_df['m'] == m_val]
    ax.errorbar(subset['g'], subset['EC_mean'], yerr=subset['EC_std'], 
                marker='s', label=f'm = {m_val}', capsize=5, linewidth=2, markersize=8)

ax.set_yscale('log')
ax.set_xlabel('Parameter $g$')
ax.set_ylabel('Number of Eulerian Circuits')
ax.legend(title='Total Edges ($m$)')
plt.tight_layout()
plt.savefig('ec_vs_g.png', dpi=300, bbox_inches='tight')
plt.close()

print("图表已成功生成：'time_vs_g.png' 和 'ec_vs_g.png'")