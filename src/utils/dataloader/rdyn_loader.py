import os.path
import pickle
import re

import networkx as nx

from src.utils.dataloader.dataloader import DataLoader


# 只支持读取snapshot或temporal格式的data.pkl数据
class RDynLoader(DataLoader):
    def __init__(self):
        super().__init__()

    def load_data(self, data_path, **kwargs):
        with open(data_path, 'rb') as f:
            return pickle.load(f)


class RDynCPDLoader(DataLoader):
    def __init__(self):
        super().__init__()

    def load_data(self, data_path: str, **kwargs):
        # load data.pkl
        with open(os.path.join(data_path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)

        if data_path.endswith('/'):
            data_path = data_path[:-1]
        if data_path.endswith('-snapshot'):
            # 因为以前做数据集的时候把weights搞错了，没有按照snapshot分开，所以在这里进行额外的处理
            # rdyn的数据不需要高于1的weight，所以直接全部默认弄成1就行了
            tmp_weights = []
            for i in range(len(data['edge_list'])):
                tmp_weights.append([1] * len(data['edge_list'][i]))
            data['weights'] = tmp_weights

        if data_path.endswith('-temporal') or data_path.endswith('-snapshot'):
            data_path = data_path[:-9]

        # 计算有多少个snapshot
        num_time_step = len(data['weights'])

        # load ground truth
        f = open(os.path.join(data_path, 'events.txt'), 'r')
        f.readline()  # first line
        cp_time_step = 1
        cp_ground_truth_snapshot = []
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if '[' in line:  # 这一行存在某种操作
                cp_ground_truth_snapshot.append(cp_time_step)
            else:  # 这一行是数字
                cp_time_step = int(re.findall('\d+', line)[0])
        f.close()
        cp_ground_truth_snapshot = {
            1: cp_ground_truth_snapshot
        }

        # load time_step -> snapshot_id的映射关系
        f = open(os.path.join(data_path, 'interactions.txt'))
        time_step2snapshot_id = {}
        last_time_step = -1
        sn_idx = 0
        for i, line in enumerate(f):
            sn_idx, _, _, _, _ = line.split('\t')
            time_step = i + 1
            # 为什么需要+1， 例如：当snapshot=5的事件全部发生完了之后，认为发生演化，此时的预测结果是6的时候认为正确
            sn_idx = int(sn_idx) + 1
            time_step = int(time_step)
            if time_step - last_time_step != 0:
                for m in range(last_time_step + 1, time_step):
                    time_step2snapshot_id[m] = sn_idx
            time_step2snapshot_id[time_step] = sn_idx
            last_time_step = time_step
        # 补全
        for time_step in range(max(time_step2snapshot_id.keys()) + 1, num_time_step):
            time_step2snapshot_id[time_step] = sn_idx

        f.close()

        return data, cp_ground_truth_snapshot, time_step2snapshot_id


class ReconstructedRDynCPDLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.rdyn_cpd_loader = RDynCPDLoader()

    def load_data(self, data_path: str, **kwargs):
        data, \
            cp_ground_truth_snapshot, \
            time_step2snapshot_id = self.rdyn_cpd_loader.load_data(data_path=data_path, **kwargs)

        edges = []  # 初始化edges列表
        last_snapshot_id = None  # 初始化当前节点编号
        tmp_edges = []

        if data_path.endswith('/'):
            data_path = data_path[:-1]
        if data_path.endswith('-temporal') or data_path.endswith('-snapshot'):
            data_path = data_path[:-9]

        # 打开文件并按行读取
        with open(os.path.join(data_path, 'interactions.txt')) as f:
            lines = f.readlines()

        edge_list_new = []
        type_list_new = []

        tmp_edge_list = []  # 这个用于构造新的edge_list_new，所以只存一个snapshot内的新边
        tmp_type_list = []  # 同上，是type_list_new的元素

        # 遍历文件中的每一行
        for i in range(len(lines)):
            line = lines[i].strip().split('\t')  # 分隔每一行中的元素为列表

            # 取出当前行的节点编号
            snapshot_id = int(line[0])
            # snapshot_id = i

            # 如果当前节点编号和之前处理的节点编号不同，则需要将之前的tmp_edges添加到edges中
            if last_snapshot_id is None or snapshot_id != last_snapshot_id:
                if last_snapshot_id is not None:
                    for _ in range(snapshot_id - last_snapshot_id):
                        edges.append(tmp_edges.copy())  # 将tmp_edges添加到edges中
                        edge_list_new.append(tmp_edge_list)
                        type_list_new.append(tmp_type_list)
                        # 清空
                        tmp_edge_list = []
                        tmp_type_list = []
                last_snapshot_id = snapshot_id  # 更新当前节点编号

            # 取出第3~5列的值，保存为一个列表
            tmp_edge = [line[2], int(line[3]), int(line[4])]
            tmp_edges.append(tmp_edge)  # 将tmp_edge添加到tmp_edges中
            tmp_edge_list.append((int(line[3]), int(line[4])))
            tmp_type_list.append(line[2])

        # 将最后一个tmp_edges添加到edges中
        edges.append(tmp_edges.copy())
        edge_list_new.append(tmp_edge_list)
        type_list_new.append(tmp_type_list)

        # 用edges列表构建图网络列表
        # edges是列表的列表
        networks = []
        for i in range(len(edges)):
            G = nx.Graph()
            for j in range(len(edges[i])):
                edge = edges[i][j]
                if edge[0] == '+':
                    G.add_edge(edge[1], edge[2])
                elif edge[0] == '-':
                    G.remove_edge(edge[1], edge[2])
            networks.append(G)
        data['graphs_new'] = networks
        data['edge_list'] = edge_list_new
        data['type_list'] = type_list_new

        return data, cp_ground_truth_snapshot, time_step2snapshot_id


class ForcedSnapshotRDynLoader(DataLoader):
    """
    强制将temporal的数据转换为snapshot版（每个时间步都创建一个snapshot），
    为了跟snapshot方法进行对比
    """

    def __init__(self):
        super().__init__()

    def load_data(self, data_path, **kwargs):
        # load data.pkl
        with open(os.path.join(data_path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)

        if data_path.endswith('/'):
            data_path = data_path[:-1]
        if not data_path.endswith('-snapshot'):
            raise RuntimeError('Please use snapshot data. '
                               'If your data is of our acceptable type, '
                               'please change the folder name to `xxx-snapshot`.')
        data_path = data_path[:-9]

        # 计算有多少个时间步
        num_time_step = len(data['weights'])

        # load ground truth
        f = open(os.path.join(data_path, 'events.txt'), 'r')
        f.readline()  # first line
        cp_time_step = 1
        cp_ground_truth_snapshot = []
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if '[' in line:  # 这一行存在某种操作
                cp_ground_truth_snapshot.append(cp_time_step)
            else:  # 这一行是数字
                cp_time_step = int(re.findall('\d+', line)[0])
        f.close()

        # load time_step -> snapshot_id的映射关系
        # 加载snapshot_id -> time_step的影射关系
        f = open(os.path.join(data_path, 'interactions.txt'))
        snapshot_id2time_step = {}
        last_time_step = -1
        sn_idx = 0
        graphs = []
        G = nx.DiGraph()
        for i, line in enumerate(f):
            sn_idx, _, relation_type, node1, node2 = line.split('\t')
            time_step = i + 1
            # 为什么需要+1， 例如：当snapshot=5的事件全部发生完了之后，认为发生演化，此时的预测结果是6的时候认为正确
            sn_idx = int(sn_idx) + 1
            time_step = int(time_step)
            if time_step - last_time_step != 0:
                for m in range(last_time_step + 1, time_step):
                    snapshot_id2time_step[sn_idx] = m
            snapshot_id2time_step[sn_idx] = time_step
            last_time_step = time_step
            if relation_type == '+':
                G.add_edge(node1, node2)
            elif (node1, node2) in G.edges:
                G.remove_edge(node1, node2)
            graphs.append(G.copy())
        # 补全
        for time_step in range(max(snapshot_id2time_step.keys()) + 1, num_time_step):
            snapshot_id2time_step[sn_idx] = time_step

        f.close()

        # 制作一个timestep版的ground truth
        cp_ground_truth_snapshot = [
            snapshot_id2time_step[s] for s in cp_ground_truth_snapshot
        ]

        cp_ground_truth_snapshot = {
            1: cp_ground_truth_snapshot
        }
        data['graphs_new'] = graphs
        return data, cp_ground_truth_snapshot, snapshot_id2time_step
