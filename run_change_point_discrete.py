# %%
import os
import sys

import src.models.components as d
import src.utils.matrix_diff_utils as mdu
from src.baselines.change_point import LAD, CICPD
from src.models.mdcpd import MDCPD
from src.utils.dataloader.rdyn_loader import ReconstructedRDynCPDLoader

sys.modules['CHANGE_POINT_INTEGRATION'] = False

# change point integration
mdu.enable_change_point()

# %%
# --- 更改参数 ---
p_renew = 0.8
sigma = 0.6
case = 1
# --- 仅更改以上参数 ---

# 以下参数不变
avg_degree = 4
paction = 0.7

# 一些超参数
MAX_PROP_ITER = 2
MIN_PROP_ITER = 1
MIN_LOSS = 1e-6
MIN_AT = 0.4
DECAY_ALPHA = 0.
WEIGHT_SHIFT = 0

data_name = 'case' + str(case) + '_500_100_4_' + str(sigma) + '_' + str(p_renew) + '_0.2_1' + '-snapshot'

rdyn_loader = ReconstructedRDynCPDLoader()

rdyn_data, target, time_step2snapshot_id = rdyn_loader.load_data(os.path.join('data', data_name))

num_cp = len(target[1])


def process_rdyn_for_lad():
    graphs_directed = []
    for g in rdyn_data['graphs_new']:
        graphs_directed.append(g.to_directed())

    return graphs_directed


# %%
# 运行LAD
lad = LAD('')
res = lad.execute(data_name, data=process_rdyn_for_lad(), including_eval=True, num_eigen=1, target=target,
                  num_nodes=rdyn_data['num_nodes'])
pred = res['prediction']
print(res)

print(lad.evaluate(margin=100, prediction=pred, target=target))
print(lad.evaluate(margin=50, prediction=pred, target=target))
print(lad.evaluate(margin=20, prediction=pred, target=target))
print(lad.evaluate(margin=5, prediction=pred, target=target))
print(lad.evaluate(margin=3, prediction=pred, target=target))
print(lad.evaluate(margin=2, prediction=pred, target=target))
print(lad.evaluate(margin=1, prediction=pred, target=target))

# %%
# 运行MDCPD-离散
md_cpd = MDCPD()
distance_function = d.ManhattanDistance()  # 采用什么距离函数

rdyn_data['num_nodes'] = 500
# turn on optimization
res = md_cpd.execute(data_name,
                     data=rdyn_data,
                     optimize=True,
                     num_cp=num_cp,
                     distance_function=distance_function)
pred = res['prediction']
print(res)

pred_snapshot = []
for ts in pred:
    pred_snapshot.append(time_step2snapshot_id[ts])
print(pred_snapshot)

print(md_cpd.evaluate(margin=5, prediction=pred_snapshot, target=target))
print(md_cpd.evaluate(margin=3, prediction=pred_snapshot, target=target))
print(md_cpd.evaluate(margin=2, prediction=pred_snapshot, target=target))
print(md_cpd.evaluate(margin=1, prediction=pred_snapshot, target=target))

# %%
# 关闭矩阵干预策略
md_cpd_raw = MDCPD()
distance_function = d.ManhattanDistance()
rdyn_data['num_nodes'] = 500

res = md_cpd_raw.execute(data_name,
                         data=rdyn_data,
                         optimize=False,  # 关闭优化，True则启用
                         num_cp=num_cp,
                         distance_function=distance_function)
pred = res['prediction']
print(res)
pred_snapshot = []
for ts in pred:
    pred_snapshot.append(time_step2snapshot_id[ts])
print(pred_snapshot)

print(md_cpd_raw.evaluate(margin=5, prediction=pred_snapshot, target=target))
print(md_cpd_raw.evaluate(margin=3, prediction=pred_snapshot, target=target))
print(md_cpd_raw.evaluate(margin=2, prediction=pred_snapshot, target=target))
print(md_cpd_raw.evaluate(margin=1, prediction=pred_snapshot, target=target))

# %%
# run CICPD on RDyn

cicpd = CICPD()
cicpd.process_rdyn_data(data_name=data_name, num_nodes=rdyn_data['num_nodes'])
res = cicpd.execute(data=rdyn_data, including_eval=True,
                    data_name=data_name, target=target, node_importance='LeaderRank',
                    flag1='1', flag2='1', kernel_mode='js')

pred = res['prediction']
print(res)

print(cicpd.evaluate(margin=5, prediction=pred, target=target))
print(cicpd.evaluate(margin=3, prediction=pred, target=target))
print(cicpd.evaluate(margin=2, prediction=pred, target=target))
print(cicpd.evaluate(margin=1, prediction=pred, target=target))
