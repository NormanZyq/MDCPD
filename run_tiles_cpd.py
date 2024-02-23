# %%
import gzip
import itertools
# 判断.gz文件是否为空
import os
import re

import numpy as np
import pandas as pd
import tiles as t

import src.metrics.cpd_metrics as m
from src.utils.dataloader.rdyn_loader import ReconstructedRDynCPDLoader, ForcedSnapshotRDynLoader


def check_and_save_non_empty_gz_files(directory_path):
    # 获取目录中所有文件的列表
    all_files = os.listdir(directory_path)

    # 筛选出strong和graph开头的文件
    strong_files = [file for file in all_files if file.startswith('strong')]
    graph_files = [file for file in all_files if file.startswith('graph')]
    # 将它们删除
    for file in strong_files:
        os.remove(os.path.join(directory_path, file))
    for file in graph_files:
        os.remove(os.path.join(directory_path, file))

    # 重新获取一次
    all_files = os.listdir(directory_path)

    # 筛选出剩余的.gz文件
    gz_files = [file for file in all_files if file.endswith('.gz') and not file.startswith('strong')]

    # 保存非空文件的文件名
    non_empty_files = []

    for gz_file in gz_files:
        file_path = os.path.join(directory_path, gz_file)

        # 使用gzip库打开.gz文件
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            # 读取文件内容
            file_content = f.read()

            # 如果文件内容非空，则将文件名保存到列表中
            if file_content:
                non_empty_files.append(gz_file)
            else:
                # 删除空文件
                os.remove(file_path)
    os.remove('./extraction_status.txt')  # 删除中间状态记录文件
    return non_empty_files


def get_tiles_cpd_result():
    # 指定目标目录路径
    target_directory = './'

    # 调用函数获取非空文件列表
    result = check_and_save_non_empty_gz_files(target_directory)
    return result


# prepare data for TILES
def create_tmp_file_for_tiles(data):
    use_snapshot = -1
    # generate a temp file for tiles
    right = use_snapshot + 1 if use_snapshot != -1 else len(data['edge_list'])
    edge_lists = data['edge_list'][: right]
    edges = list(itertools.chain.from_iterable(edge_lists))
    node1 = [edge[0] for edge in edges]
    node2 = [edge[1] for edge in edges]
    type_lists = data['type_list'][: right]
    type_list = list(itertools.chain.from_iterable(type_lists))
    timestamps = np.arange(1, len(node1) * 86400 + 1, 86400)
    df = pd.DataFrame({
        'action': type_list,
        'n1': node1,
        'n2': node2,
        'timestamps': timestamps,
    })
    # tmp_name = '/home/zhuyeqi/zhuyeqi_from_118/DyLPA/data/tiles/tmp/tmp.txt'
    df.to_csv(tmp_name, sep='\t', index=False, header=False)


# run tiles
def run_tiles():
    tl = t.eTILES(tmp_name, obs=1)
    tl.execute()


# %%
# 需要变化的参数
p_renew = 0.6
sigma = 0.6

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

# %%
rdyn_loader = ReconstructedRDynCPDLoader()

rdyn_data, target, time_step2snapshot_id = rdyn_loader.load_data(os.path.join('data', data_name))

num_cp = len(target[1])
data = rdyn_data
tmp_name = './data/tiles/tmp/tmp.txt'

# %%
# 离散场景运行tiles
create_tmp_file_for_tiles(data)
run_tiles()
result = get_tiles_cpd_result()

# 遍历result，其每个元素的格式为`xxxx-数字.gz`，现将数字提取出来
pred = [int(re.findall(r'\d+', x)[0]) for x in result]

# 根据此前的映射将temporal id转换为snapshot id
pred_snapshot = [time_step2snapshot_id[x] for x in pred]

print(pred_snapshot)


# %%

# 创建一个临时的评价函数
def evaluate_temp(target, pred, margin):
    f1, p, r = m.cpd_f1_with_margin(target, pred, margin, return_PR=True)
    delta_t = m.delta_t(pred, target)

    return f1, p, r, delta_t


# %%
print('离散时间结果')
print(evaluate_temp(target, pred_snapshot, 10))
print(evaluate_temp(target, pred_snapshot, 5))
print(evaluate_temp(target, pred_snapshot, 2))
print(evaluate_temp(target, pred_snapshot, 1))

# %%
# 在连续时间场景上做相同实验
# load data
data_name = 'case' + str(case) + '_500_100_4_' + str(sigma) + '_' + str(p_renew) + '_0.2_1' + '-snapshot'

loader = ForcedSnapshotRDynLoader()
data, target, snapshot_id2time_step = loader.load_data(os.path.join('data', data_name))
tmp_name = './data/tiles/tmp/tmp.txt'
create_tmp_file_for_tiles(data)

# %%
run_tiles()
result = get_tiles_cpd_result()

# 遍历result，其每个元素的格式为`xxxx-数字.gz`，现将数字提取出来
pred = [int(re.findall(r'\d+', x)[0]) for x in result]

# %%
print('连续时间结果')
print(evaluate_temp(target, pred, 100))
print(evaluate_temp(target, pred, 50))
print(evaluate_temp(target, pred, 20))
print(evaluate_temp(target, pred, 5))

# %%
