# %%
import numpy as np

from src.dy_lpa import DyLPA
from src.metrics.cd_metrics import normalized_mutual_information
from src.utils.utils import DataTool
import matplotlib.pyplot as plt

# %%
MAX_PROP_ITER = 20
MIN_PROP_ITER = 20
MIN_AT = 1.0
WEIGHT_SHIFT = 1
MIN_LOSS = 1e-4
GATE = 3e-4
DECAY_ALPHA = 0.09
# level = 0

dt = DataTool('connected_data_final', weight_shift=WEIGHT_SHIFT, weight_normalize=False)
data, weights = dt.get_data()
partition_true, communities_true = dt.get_ground_truth()
NUM_NODES = data['num_nodes']
NUM_COMMUNITIES = NUM_NODES
# %%
# 0.025 版本已经获取的检测结果直接拿过来，用于画图分析
evo_step = [8, 188, 190, 192, 263, 289, 304, 308, 386, 405, 490,
            548, 579, 657, 683, 759, 809, 823, 841, 935, 940, 949,
            967, 1131, 1264, 1281, 1317, 1486, 1490, 1496, 1541, 1625, 1681,
            1718, 1759, 1809, 1822, 2038, 2040, 2300, 2302, 2305, 2375, 2405,
            2409, 2520, 2629, 2678, 2710, 2734, 2889, 2905, 2986, 3054, 3158,
            3226, 3436, 3504, 3573, 3613, 3651, 3673, 3854, 4016, 4061, 4129,
            4356, 4358, 4419, 4422, 4428, 4857, 5266, 5291, 5293, 5537, 5935,
            6082, 6183, 6435, 6618, 7246, 7468]

max_idx = 10431

dl = DyLPA(NUM_NODES, MAX_PROP_ITER,
           min_iter=MIN_PROP_ITER,
           min_loss=MIN_LOSS,
           decay_alpha=DECAY_ALPHA,
           min_at=MIN_AT)

# %%
left = 0
# F_before = {}
# F_after = {}
# A_before = {}
# A_after = {}
g_before = {}
g_after = {}
pred_before = {}
pred_after = {}
for e in evo_step:
    # 先加到e（不包括），取得一次F，是before
    dl.add_edges(data['edge_list'][left:e],
                 weights[left: e],
                 check_interval=0)
    p, c = dl.predict(level=3)
    pred_before[e] = (p, c)
    # F_before[e] = np.array(dl.F)
    # A_before[e] = dl.A.copy()
    g_before[e] = dl.G.copy()
    right = min(e + 1, max_idx)
    # 加入e本身，取得一次F，是after
    dl.add_edges(data['edge_list'][e: right],
                 weights[e: right],
                 check_interval=0)
    p, c = dl.predict(level=3)
    pred_after[e] = (p, c)
    # F_after[e] = np.array(dl.F)
    # A_after[e] = dl.A.copy()
    g_after[e] = dl.G.copy()
    left = right

# %%
# 人工打印并排查
nmis = []
for step in evo_step:
    look = step
    print('before', look, 'len =', len(pred_before[look][1]))
    print('after', look, 'len =', len(pred_after[look][1]))
    print('delta', look, len(pred_before[look][1]) - len(pred_after[look][1]))
    nmi = normalized_mutual_information(pred_after[look][1], pred_before[look][1])
    nmis.append(nmi)
    print('nmi', look, nmi)
    print('\n')
    cid = 0
    # print('before', look, 'len0 =', len(pred_before[look][1][cid]))
    # print('after', look, 'len0 =', len(pred_after[look][1][cid]))

plt.plot(nmis)
plt.show()

#%%
record = []
for step in evo_step:
    look = step
    pb, cb = pred_before[look]
    pa, ca = pred_after[look]
    tmp = []
    for node in pb:
        if pb[node] != pa[node]:
            # record.append('node {}({}) change from {} to {}'.format())
            print('node {}({}) change from {} to {} at time {}'.format(node, data['id2name'][node], pb[node], pa[node], step))
    print()

# %%
# 最粗暴手段
print(pred_before[look][1])
print(pred_after[look][1])

# %%
# 将G全部导出，用gephi打开？
import networkx as nx

for step in evo_step:
    gb = g_before[step]
    ga = g_after[step]
    nx.write_gexf(gb, 'gephi/step{}_before.gexf'.format(step))
    nx.write_gexf(ga, 'gephi/step{}_after.gexf'.format(step))

