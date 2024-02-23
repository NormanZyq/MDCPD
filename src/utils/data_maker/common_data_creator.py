# %%
import networkx as nx
import pandas as pd
import pickle
import time
from src.utils.data_utils import build_graph, largest_cc_data, graph2csv

full_data_name = 'email_eu_core'

# source_col = 'From'
# target_col = 'To'
# time_col = 'Datetime'
# weight_col = 'Duration(seconds)'
# belong_col = 'Cell Tower'
source_col = 'sender'
target_col = 'receiver'
# time_col = 'TIMESTAMP'
# weight_col = 'Duration(seconds)'
# belong_col = 'event'

# data = pd.read_excel('data/data1025.xlsx', engine='openpyxl')
data = pd.read_csv('data/{}/raw.csv'.format(full_data_name))
# pd.ra
# data.sort_values(time_col, ascending=True, inplace=True)
# 打乱email数据集
# data = data.sample(frac=1).reset_index(drop=True)

# %%
sources = list(data[source_col])
targets = list(data[target_col])
# timestamp = list(data[time_col])
# relations = list(data['relation'])
# digit = list(data[weight_col])
digit = [1] * len(sources)

#%%
id2name = [str(n) for n in list(set(sources + targets))]
name2id = {
    v: k for k, v in enumerate(id2name)
}
source_ids = [name2id[str(name)] for name in sources]
target_ids = [name2id[str(name)] for name in targets]


# %%
data_dict = {
    'id2name': id2name,
    'name2id': name2id,
    'num_nodes': len(id2name),
    'edge_list': [],
    # 'edge_relation': relations,
    'edge_digit': digit,
    # 'timestamp': timestamp,
    'timestamp': list(range(len(digit))),
    'data_create_time': time.asctime(time.localtime(time.time())),
}

for i in range(len(sources)):
    n1 = source_ids[i]
    n2 = target_ids[i]
    data_dict['edge_list'].append([n1, n2])

# %%
with open('data/{}/data.pkl'.format(full_data_name), 'wb') as f1:
    pickle.dump(data_dict, f1)

#%%
# -------------------------------------------------------------------
# 用完整数据构造最大连通图的数据
graph = nx.Graph()
# 直接用名字构造出来这个图，然后导出成csv并保持顺序
for i in range(len(sources)):
    graph.add_edge(sources[i], targets[i],
                   # timestamp=data_dict['timestamp'][i],
                   # relation=data_dict['edge_relation'][i],
                   weight=data_dict['edge_digit'][i])
largest_cc = max(nx.connected_components(graph), key=len)
largest_G = graph.subgraph(largest_cc).copy()
largest_nodes = set(largest_G.nodes)

lar_sources, lar_targets, lar_digit = [], [], []
for i in range(len(sources)):
    s, t = sources[i], targets[i]
    if s in largest_nodes and t in largest_nodes:
        lar_sources.append(s)
        lar_targets.append(t)
        # lar_relations.append(relations[i])
        # lar_timestamp.append(timestamp[i])
        lar_digit.append(digit[i])

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
    # 'edge_relation': lar_relations,
    'edge_digit': lar_digit,
    # 'timestamp': lar_timestamp,
    'data_create_time': time.asctime(time.localtime(time.time())),
}

# for i in range(len(sources)):
#     n1 = source_ids[i]
#     n2 = target_ids[i]
#     data_dict['edge_list'].append([n1, n2])
#%%
connected_data_name = 'email_connected'
with open('data/{}/data.pkl'.format(connected_data_name), 'wb') as f1:
    pickle.dump(lar_data_dict, f1)

