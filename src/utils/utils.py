import os.path as op
import pickle

import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd
import community


from src.metrics.cd_metrics import f1_components, normalized_mutual_information, normalized_f1_score


def partition2community(partition):
    """
    Convert partition type into communities type
    :param partition: dict node_id -> community_id
    :return: community list. Each element is a list of node in the same community
    """
    communities = {}
    if type(partition) == dict:
        for idx in partition:
            cid = partition[idx]
            if cid in communities:
                communities[cid].append(idx)
            else:
                communities[cid] = [idx]
    elif type(partition) == list:
        for idx, cid in enumerate(partition):
            if cid in communities:
                communities[cid].append(idx)
            else:
                communities[cid] = [idx]
    return list(communities.values())


def community2partition(communities):
    """
    Convert community type into partition type. The start id of partition is zero.
    All communities are anonymous.
    :param communities: a list：[[node0, node1, ...], [node_i, node_j, node_k, ...], ...]
                                |_________________|, |___________________________|,  |_|
                                In a same community,         Same community,       others
    :return: partition dict (node_id -> community_id)
    """
    partition = {}
    for i, c in enumerate(communities):
        for node in c:
            partition[node] = i
    return partition


def visualize(G, partition, **kwargs):
    cmap = cm.get_cmap('plasma', max(partition.values()) + 1)
    nx.draw(G, pos=nx.fruchterman_reingold_layout(G),
            nodelist=list(partition.keys()),
            # node_size=[G.degree[nid] ** 2 * 1 for nid in partition.keys()],
            node_size=kwargs.get('node_size', 30),
            cmap=cmap,
            node_color=[v for v in partition.values()], width=0.05)
    # plt.show()


def draw_nodes(G, partition):
    cmap = cm.get_cmap('plasma', max(partition.values()) + 1)
    # 生成一个社区分配p的子集，因为可能只添加了部分的边
    sub_p = {k: partition[k] for k in set(partition.keys()) & set(G.nodes)}  # 子集
    nx.draw_networkx_nodes(G, pos=nx.fruchterman_reingold_layout(G),
                           nodelist=list(sub_p.keys()),
                           # node_size=[G.degree[nid] ** 2 * 1 for nid in partition.keys()],
                           node_size=6,
                           cmap=cmap,
                           node_color=[v for v in sub_p.values()])


def normalization(data, axis=0):
    min_values = np.min(data, axis=axis)
    _range = np.max(data, axis=axis) - np.min(data, axis=axis)
    return (data - min_values) / _range


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax_value = x_exp / x_exp_row_sum
    return softmax_value


def modularity(graph: nx.Graph, cluster):
    e = 0.0
    a_2 = 0.0
    cluster_degree_table = {}
    for vtx in graph.nodes:
        adj = list(nx.neighbors(graph, vtx))
        label = cluster[vtx]
        for neighbor in adj:
            if label == cluster[neighbor]:
                e += 1
        if label not in cluster_degree_table:
            cluster_degree_table[label] = 0
        cluster_degree_table[label] += len(adj)
    e /= 2 * graph.number_of_edges()

    for label, cnt in cluster_degree_table.items():
        a = 0.5 * cnt / graph.number_of_edges()
        a_2 += a * a

    Q = e - a_2
    return Q


class DataTool:
    """
    This class is the tool to process data for LSED.
    There were many wrong annotations at first, and with this class we can alter the annotations quickly.
    So we keep this class for better compatibility. For other data, we can use `CommonDataTool`.
    """

    def __init__(self, data, load_annotation=True, weight_shift=0, weight_normalize=True):
        """
        Initializer
        :param data: data name
        :param load_annotation:
        :param weight_shift:    `k`
        :param weight_normalize:    True if normalize weight
        """
        self.data, self.weights, self.communities = None, None, None  # init data and weights
        if type(data) == str:
            self.data_path = 'data/' + data + '/data.pkl'
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
            if weight_normalize:
                self.weights = (normalization(np.array(self.data['edge_digit']), axis=0) + weight_shift).tolist()
            else:
                self.weights = (np.array(self.data['edge_digit']) + weight_shift).tolist()
            # load community ground truth
            anno_path = 'data/' + data + '/annotation.xlsx'
            part_path = 'data/' + data + '/partition.xlsx'
            if load_annotation:
                # parse annotation of LSED ground truth
                if op.exists(anno_path):  # “共现”格式的标注
                    print('Found community annotation file! Loading...')
                    # load file
                    annotation = pd.read_excel(anno_path, usecols='A:D')
                    names = list(annotation['节点名称'].astype(str))
                    auto = list(annotation['自动标注'].astype(str))
                    manual = list(annotation['人工结果'].astype(str))
                    # name->with what
                    annotation_dict = {
                        names[i]: manual[i] if auto[i] == '-1' else auto[i] for i in range(len(names))
                    }
                    communities = dict()
                    # four cases
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
                        curr_community = [self.data['name2id'][n] for n in communities[k]]
                        if curr_community not in real_communities:
                            real_communities.append(curr_community)
                    self.communities = real_communities
                    print('Community annotation loaded.')
                elif op.exists(part_path):  # partition type --> no use, comment out
                    # print('Found community annotation file! Loading...')
                    # # load file
                    # partition = pd.read_excel(part_path, usecols='A:B')
                    # names, label = list(partition['node'].astype(str)), \
                    #                       list(partition['label'].astype(int))
                    # communities = dict()
                    # for i in range(len(names)):
                    #     communities[names[i]] = label[i]
                    # real_communities = []
                    # for k in communities:
                    #     curr_community = [self.data['name2id'][n] for n in communities[k]]
                    #     if curr_community not in real_communities:
                    #         real_communities.append(curr_community)
                    # self.communities = real_communities
                    # print('Community annotation loaded.')
                    pass
        elif type(data) == list:
            # if a list is passed in, a temp data will be created for test
            transposed = list(map(list, zip(*data)))
            nodes = set(transposed[0]) | set(transposed[1])
            num_nodes = len(nodes)
            id2name = {i: n for i, n in enumerate(nodes)}
            name2id = {v: k for k, v in id2name.items()}
            self.data = {
                'edge_list': data,
                'id2name': id2name,
                'name2id': name2id,
                'num_nodes': num_nodes
            }
            self.weights = [1] * len(data)

    def get_data(self):
        return self.data, self.weights

    def get_ground_truth(self):
        if self.communities is None:
            return None, None
        return self.get_partition_true(), self.communities

    def get_partition_true(self):
        partition_true = {}
        for i, c in enumerate(self.communities):
            for nid in c:
                partition_true[nid] = i
        return partition_true

    def make_final_network(self, use_name=False) -> nx.Graph:
        """
        use data and edge list to build the final network.
        :param use_name: True if you want to use the name of entity to create the network,
                         False if you use the id as node name in the network
        :return: a graph contains all edges in the dataset. Note that the weight of edge is ignored
        """
        data, weights = self.get_data()
        edges = data['edge_list']
        temp_graph = nx.Graph()
        for i, e in enumerate(edges):
            if use_name:
                temp_graph.add_edge(data['id2name'][e[0]], data['id2name'][e[1]])
                pass
            else:
                temp_graph.add_edge(e[0], e[1])
        return temp_graph


def load_partition(part_path, scope):
    """
    Now no use. Will be removed in the future
    :param part_path:
    :param scope:
    :return:
    """
    partition = pd.read_excel(part_path, usecols='A:B')
    names, label = list(partition['node'].astype(int)), \
                   list(partition['label'].astype(int))
    communities = dict()
    for i in range(len(names)):
        if names[i] not in scope:
            continue
        communities[names[i]] = label[i]
    return communities


# 查看每个社区都有哪些节点并显示名字
def print_communities(c, id2name):
    with open('communities.txt', 'w') as f:
        for cid in sorted(c.keys()):
            f.write('------community prediction {}------\n'.format(cid))
            f.write(str([id2name[nid] for nid in c[cid]]) + '\n')


def evaluate(partition, target):
    common_nodes = set(partition.keys()) & set(target.keys())
    partition = {
        k: partition[k] for k in common_nodes
    }
    target = {
        k: target[k] for k in common_nodes
    }
    # padding process
    com_prediction, com_true = partition2community(partition), partition2community(target)
    # print('len(com_prediction)={}'.format(len(com_prediction)))
    # print('len(com_true)={}'.format(len(com_true)))
    tp, tn, fp, fn = f1_components(partition, target)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    f1_norm = normalized_f1_score(partition, target)
    # Normalized mutual information
    nmi = normalized_mutual_information(com_true, com_prediction)
    evaluation = {
        'Num prediction': len(com_prediction),
        'Num ground truth': len(com_true),
        'NMI': nmi,
        'P': precision,
        'R': recall,
        'Acc': acc,
        'F1': f1,
        'NF1': f1_norm
        # 'Num Communities Pred': len(com_prediction),
        # 'Num Communities True': len(com_true)
    }
    return evaluation
