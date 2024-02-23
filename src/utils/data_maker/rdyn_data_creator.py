#%%
import os
import os.path as op
import pickle
import time
import re

import networkx as nx
import pandas as pd

# %%
# RDyn数据集
# 读取
# data_dir = 'data/10000_100_4_0.6_0.8_0.2_1'
data_dir = 'data/case2_500_100_4_0.7_0.8_0.2_1-significance-3'
data_path = op.join(data_dir, 'interactions.txt')

# 读取交互列表
interactions = pd.read_csv(data_path, sep='\t', names=['iteration_id', 'interaction_id', 'type', 'node1', 'node2'])

# %%
# 按照iteration读取数据集，可理解为snapshot
sources_list = []
targets_list = []
timestamps_list = []
iteration2edges = {}
iteration2types = {}
interaction_id_list = []
snapshot_id_list = []
for group in interactions.groupby('iteration_id'):
    iter_id = group[0]
    dataframe = group[1]
    snapshot_id_list.append(list(dataframe['iteration_id']))
    sources = list(dataframe['node1'])
    targets = list(dataframe['node2'])
    sources_list.append(sources)
    targets_list.append(targets)
    # types_list.append(list(dataframe['type']))
    timestamps_list.append(list(dataframe['interaction_id']))
    curr_edges = [i for i in zip(sources, targets)]
    curr_types = list(dataframe['type'])
    # edges_list.append(curr_edges)
    iteration2edges[iter_id] = curr_edges
    iteration2types[iter_id] = curr_types

# 读取ground truth和graph文件
community_true_list = []
partition_true_list = []
com_filenames = []
graphs = []
edges_list = []
types_list = []
last_iter_id = 0
files = os.listdir(data_dir)
# for n in sorted(os.listdir(data_dir)):
for file_id in range(100):
    com_file = 'communities-{}.txt'.format(file_id)
    graph_file = 'graph-{}.txt'.format(file_id)
    if com_file in files:
        with open(op.join(data_dir, com_file), 'r') as f:
            cur_coms = []
            cur_partition = {}
            for line in f:
                parts = line.split('\t')
                coms = eval(parts[1])
                cur_coms.append(coms)
                for nid in coms:
                    cur_partition[nid] = int(parts[0])
            community_true_list.append(cur_coms)
            partition_true_list.append(cur_partition)
    if graph_file in files:
        g_edges = pd.read_csv(op.join(data_dir, graph_file), sep='\t', names=['source', 'target'], dtype=int)
        g_edges = zip(g_edges['source'], g_edges['target'])
        # 读取数值
        num = file_id
        # print(num)
        temp_edges = []
        temp_types = []
        for i in range(last_iter_id, num + 1):
            # print(i)
            temp_edges.extend(iteration2edges.get(i, []))       # iteration从0开始，但是graph从1开始
            temp_types.extend(iteration2types.get(i, []))
        edges_list.append(temp_edges)
        types_list.append(temp_types)
        last_iter_id = num + 1
        g = nx.Graph()
        g.add_edges_from(g_edges)
        graphs.append(g)

# %%
# 全量数据
sources_all = list(interactions['node1'])
targets_all = list(interactions['node2'])
types_all = list(interactions['type'])
timestamps_all = list(interactions['interaction_id'])
iteration_id_all = list(interactions['iteration_id'])
edges_all = [i for i in zip(sources_all, targets_all)]

# %%
id2name = [n for n in list(set(sources_all + targets_all))]
name2id = {
    v: k for k, v in enumerate(id2name)
}
source_ids = [name2id[name] for name in sources_all]
target_ids = [name2id[name] for name in targets_all]

# %%
# 导出一份我的格式的数据集
data_name = data_dir + '-snapshot'

data_dict = {
    'id2name': id2name,
    'name2id': name2id,
    'num_nodes': len(id2name),
    'snapshot_id': snapshot_id_list,  # 这个字段表示这一条边属于第几个snapshot
    'graphs': graphs,
    'edge_list': edges_list,
    'weights': [1] * len(types_all),
    'type_list': types_list,
    'timestamps': timestamps_list,
    'data_create_time': time.asctime(time.localtime(time.time())),
    'partition': partition_true_list,
    'communities': community_true_list,
}

if not op.exists(data_name):
    os.mkdir(data_name)

with open(op.join(data_name, 'data.pkl'), 'wb') as f1:
    pickle.dump(data_dict, f1)

# %%
data_name = data_dir + '-temporal'

data_dict = {
    'id2name': id2name,
    'name2id': name2id,
    'num_nodes': len(id2name),
    'snapshot_id': iteration_id_all,  # 这个字段表示这一条边属于第几个snapshot
    'edge_list': [[n1, n2] for n1, n2 in edges_all],
    'weights': [1] * len(edges_all),
    'type_list': types_all,
    'timestamps': timestamps_all,
    'data_create_time': time.asctime(time.localtime(time.time())),
    'partition': partition_true_list[-1],
    'communities': community_true_list[-1],
}

# %%
if not op.exists(data_name):
    os.mkdir(data_name)

with open(op.join(data_name, 'data.pkl'), 'wb') as f1:
    pickle.dump(data_dict, f1)
