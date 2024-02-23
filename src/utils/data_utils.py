import os.path as op
import pickle
import random
import time

import networkx as nx
import pandas as pd


# def baseline_load_data():
#     data_path = '/home/zhuyeqi/DyLPA/data/rdyn_positive_data0323/data.pkl'
#     with open(data_path, 'rb') as f:
#         data = pickle.load(f)
#     return data, data['partition'], data['communities']


def graph2csv(graph: nx.Graph, random_time=False):
    edges = list(graph.edges)
    df = pd.DataFrame(edges)
    if random_time:
        pd.concat((df, pd.DataFrame([random.random() for _ in range(df.shape[0])], columns=['time'])), axis=1)
    df.sort_values(by='time', inplace=True)
    return df


def build_graph(data):
    weights = data['weights']
    edges = data['edge_list']
    temp_graph = nx.Graph()
    for i, e in enumerate(edges):
        temp_graph.add_edge(e[0], e[1], weight=weights[i])
    return temp_graph


def largest_cc_data(graph: nx.Graph, data):
    """
    将graph转换为最大连通图，并按照data的格式返回
    注：节点的编号不会变
    """
    largest_cc = max(nx.connected_components(graph), key=len)
    largest_G = graph.subgraph(largest_cc).copy()
    largest_nodes = set(largest_G.nodes)

    use_index = []
    for i, edge in enumerate(data['edge_list']):
        n1 = edge[0]
        n2 = edge[1]
        if n1 in largest_nodes and n2 in largest_nodes:
            use_index.append(i)

    data_largest = {k: [] for k in data.keys()}
    data_largest['id2name'] = dict([(key, data['id2name'][key]) for key in largest_nodes])
    data_largest['name2id'] = dict([(key, data['name2id'][key]) for key in data_largest['id2name'].values()])

    for i in use_index:
        data_largest['edge_list'].append(data['edge_list'][i])
        data_largest['edge_relation'].append(data['edge_relation'][i])
        data_largest['edge_digit'].append(data['edge_digit'][i])
        data_largest['timestamp'].append(data['timestamp'])
    data_largest['num_nodes'] = len(data_largest['id2name'])
    data_largest['data_create_time'] = time.asctime(time.localtime(time.time()))

    return largest_G, data_largest


class CommonDataTool:
    def __init__(self, data_name, weight_shift=0):
        if not op.isdir(op.join('data', data_name)):
            raise FileNotFoundError('Not found ' + data_name + ' data set')
        with open(op.join('data', data_name, 'data.pkl'), 'rb') as file:
            self.data = pickle.load(file)
            if weight_shift:
                self.data['weights'] = [w + weight_shift for w in self.data['weights']]

    def get_data(self):
        return self.data, self.data['partition'], self.data['communities']

    def make_final_network(self):
        return build_graph(self.data)
