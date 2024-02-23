"""
Updated: 6.22, 2022
Will create two versions of LSED: snapshot and temporal
snapshot's `edge_list` is a list of edge list in one snapshot
"""


# %%
import math
import pickle
import time
import os
import networkx as nx
import pandas as pd

full_data_name = 'full_data_huixin'
# data = pd.read_excel('data/data1025.xlsx', engine='openpyxl')
data = pd.read_csv('data/{}/raw.csv'.format(full_data_name))
data.sort_values('time', ascending=True, inplace=True)

# %%
sources = list(data['source'])
targets = list(data['target'])
timestamp = list(data['time'])
relations = list(data['relation'])
digit = list(data['relation_digit'])

id2name = list(set(sources + targets))
name2id = {
    v: k for k, v in enumerate(id2name)
}
edges = [[name2id[name1], name2id[name2]] for name1, name2 in list(zip(sources, targets))]

# %%
data_dict = {
    'id2name': id2name,
    'name2id': name2id,
    'num_nodes': len(id2name),
    'edge_list': edges,
    'edge_relation': relations,
    'edge_digit': digit,
    'timestamp': timestamp,
    'data_create_time': time.asctime(time.localtime(time.time())),
}

# %%
with open('data/{}/data.pkl'.format(full_data_name), 'wb') as f1:
    pickle.dump(data_dict, f1)

# %%
# -------------------------------------------------------------------
# 用完整数据构造最大连通图的数据
connected_data_name = 'connected_data_huixin'

graph = nx.Graph()
# 直接用名字构造出来这个图，然后导出成csv并保持顺序
for i in range(len(sources)):
    graph.add_edge(sources[i], targets[i],
                   timestamp=data_dict['timestamp'][i],
                   relation=data_dict['edge_relation'][i],
                   weight=data_dict['edge_digit'][i])
largest_cc = max(nx.connected_components(graph), key=len)
largest_G = graph.subgraph(largest_cc).copy()
largest_nodes = set(largest_G.nodes)

lar_sources, lar_targets, lar_relations, lar_digit, lar_timestamp = [], [], [], [], []
for i in range(len(sources)):
    s, t = sources[i], targets[i]
    if s in largest_nodes and t in largest_nodes:
        lar_sources.append(s)
        lar_targets.append(t)
        lar_relations.append(relations[i])
        lar_digit.append(digit[i])
        lar_timestamp.append(timestamp[i])

lar_id2name = list(set(lar_sources + lar_targets))
lar_name2id = {
    v: k for k, v in enumerate(lar_id2name)
}
lar_source_ids = [lar_name2id[name] for name in lar_sources]
lar_target_ids = [lar_name2id[name] for name in lar_targets]
# 制作数据
lar_data_dict = {
    'id2name': lar_id2name,
    'name2id': lar_name2id,
    'num_nodes': len(lar_id2name),
    'edge_list': list(zip(lar_source_ids, lar_target_ids)),
    'edge_relation': lar_relations,
    'edge_digit': lar_digit,
    'timestamp': lar_timestamp,
    'data_create_time': time.asctime(time.localtime(time.time())),
}

# %%
# 读取标注
# load file
anno_path = 'data/' + connected_data_name + '/annotation.xlsx'
annotation = pd.read_excel(anno_path, usecols='A:D')
names, auto, manual = list(annotation['节点名称'].astype(str)), \
                      list(annotation['自动标注'].astype(str)), \
                      list(annotation['人工结果'].astype(str))
# name->with what
annotation_dict = {
    names[i]: manual[i] if auto[i] == '-1' else auto[i] for i in range(len(names))
}
communities = dict()
for n in annotation_dict:
    if n in communities and annotation_dict[n] not in communities:  # 自己已经在了但是目标不在
        target_community = communities[n]
        target_community.add(annotation_dict[n])
        for nid in target_community:
            communities[nid] = target_community
    elif n in communities and annotation_dict[n] in communities:  # 自己和目标都在
        target_community = communities[n] | communities[annotation_dict[n]]  # 取并集
        for nid in target_community:
            communities[nid] = target_community
    elif n not in communities and annotation_dict[n] in communities:  # 自己不在目标在
        target_community = communities[annotation_dict[n]]
        target_community.add(n)
        for nid in target_community:
            communities[nid] = target_community
    else:  # 最后一种情况：都不在
        temp_community = {n, annotation_dict[n]}
        communities[n] = communities[annotation_dict[n]] = temp_community
real_communities = []
for k in communities:
    curr_community = [lar_data_dict['name2id'][n] for n in communities[k]]
    if curr_community not in real_communities:
        real_communities.append(curr_community)
colors = list(range(len(real_communities)))
color_dict = {}
for i, nodes in enumerate(real_communities):
    for n in nodes:
        color_dict[data['name2id'][n]] = colors[i]
node_color_list = [color_dict[i] for i in range(len(annotation_dict))]
lar_data_dict['partition'] = color_dict
lar_data_dict['communities'] = real_communities

# %%
with open('data/{}/data.pkl'.format(connected_data_name), 'wb') as f1:
    pickle.dump(lar_data_dict, f1)

# %%
# 直接读取connected的pkl，
# 然后创建一个snapshot数据，
# 每个snapshot有1000条边，最后那个snapshot有多少是多少
connected_data_name = 'connected_data_huixin'  # 直接把上面的这句代码复制过来方便使用而已

with open('data/{}/data.pkl'.format(connected_data_name), 'rb') as f:
    connected_data = pickle.load(f)

edges = connected_data['edge_list']
full_edge_list_in_snapshot = []  # 每个元素都是一个snapshot，保存该snapshot的全量edge
increment_edge_list_in_snapshot = []  # 增量
for snap_id in range(1, math.ceil(len(edges) / 1000) + 1):
    full_edge_list_in_snapshot.append(edges[:min(snap_id * 1000, len(edges))])
    increment_edge_list_in_snapshot.append(edges[(snap_id - 1) * 1000: min(snap_id * 1000, len(edges))])

graphs = []  # 制作图网络
for e_list in full_edge_list_in_snapshot:
    temp_graph = nx.Graph()
    # 为了实现权重的变化，于是手工添加
    for e in e_list:
        if temp_graph.has_edge(e[0], e[1]):
            temp_graph.add_edge(e[0], e[1], weight=temp_graph.get_edge_data(e[0], e[1])['weight'] + 1)
        else:
            temp_graph.add_edge(e[0], e[1], weight=1)
    graphs.append(temp_graph)

connected_data['full_edge_list_in_snapshot'] = full_edge_list_in_snapshot
connected_data['increment_edge_list_in_snapshot'] = increment_edge_list_in_snapshot
connected_data['graphs'] = graphs
connected_data['data_create_time'] = time.asctime(time.localtime(time.time())),
connected_data['partition'] = [partition_true for _ in range(len(graphs))]
connected_data['communities'] = [communities_true for _ in range(len(graphs))]

if not os.path.exists('data/{}-snapshot'.format(connected_data_name)):
    os.mkdir('data/{}-snapshot'.format(connected_data_name))
with open('data/{}-snapshot/data.pkl'.format(connected_data_name), 'wb') as f1:
    pickle.dump(connected_data, f1)
