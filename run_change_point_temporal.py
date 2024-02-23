# %%
import os
import sys

import src.models.components as d
import src.utils.matrix_diff_utils as mdu
from src.models.mdcpd import MDCPD, MDCPDLargeScaleCompatibleVersion
from src.utils.dataloader.lsed_loader import LSEDTemporalDataLoader
from src.utils.dataloader.rdyn_loader import ForcedSnapshotRDynLoader

sys.modules['CHANGE_POINT_INTEGRATION'] = False

# change point integration
mdu.enable_change_point()

# %%
p_renew = 0.8
sigma = 0.6
approach = 'DyLPA'
case = 1

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

rdyn_loader = ForcedSnapshotRDynLoader()

rdyn_data, target, time_step2snapshot_id = rdyn_loader.load_data(os.path.join('data', data_name))

num_cp = len(target[1])

# %%
# 运行MDCPD
md_cpd = MDCPD()
distance_function = d.ManhattanDistance()

rdyn_data['num_nodes'] = 500
# turn on optimization
res = md_cpd.execute(data_name,
                     data=rdyn_data,
                     optimize=True,
                     num_cp=num_cp,
                     distance_function=distance_function)
pred = res['prediction']
print(res)

pred_snapshot = pred

print(md_cpd.evaluate(margin=100, prediction=pred_snapshot, target=target))
print(md_cpd.evaluate(margin=50, prediction=pred_snapshot, target=target))
print(md_cpd.evaluate(margin=20, prediction=pred_snapshot, target=target))
print(md_cpd.evaluate(margin=5, prediction=pred_snapshot, target=target))

# %%
# turn off optimization
md_cpd_raw = MDCPD()
distance_function = d.ChebyshevDistance()
rdyn_data['num_nodes'] = 500

res = md_cpd_raw.execute(data_name,
                         data=rdyn_data,
                         optimize=False,
                         num_cp=num_cp,
                         distance_function=distance_function)
pred = res['prediction']
pred_snapshot = pred

print(md_cpd_raw.evaluate(margin=100, prediction=pred_snapshot, target=target))
print(md_cpd_raw.evaluate(margin=50, prediction=pred_snapshot, target=target))
print(md_cpd_raw.evaluate(margin=20, prediction=pred_snapshot, target=target))
print(md_cpd_raw.evaluate(margin=5, prediction=pred_snapshot, target=target))

# %%
# 在LSED上进行case study，然后配合case_study.py+gephi进行可视化
lsed_data_loader = LSEDTemporalDataLoader()
lsed_data_name = 'connected_data_final-snapshot'
lsed_data = lsed_data_loader.load_data(lsed_data_name)

# %%
md_cpd_temporal = MDCPDLargeScaleCompatibleVersion()
k = 0.025
num_cp = int(len(lsed_data['edge_list']) * k)
res = md_cpd_temporal.execute(data_name='LSED',
                              data=lsed_data,
                              max_prop_iter=20,
                              alpha=0.09,
                              optimize=False,
                              edge_per_snapshot=100,  # 对边进行分批返回，减少空间代价
                              num_cp=num_cp)
print(res)
