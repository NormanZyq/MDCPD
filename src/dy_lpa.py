# %%
import sys
from typing import Union

import networkx as nx
import numpy as np
from community import modularity
from tqdm import tqdm

from src.utils.utils import softmax, partition2community

# alpha = 0.09
# E_REVERSE = 1 / (np.e ** alpha)
E_REVERSE = 1

ROW_AND_COL = ['row', 'col']
ROW = ['row']
COL = ['col']


# F_records = []
# full_F_records = []
# full_D_records = []

def extend(which, extend_size: int, extend_what, fill_diag_by=0):
    """
    在行/列（0或1维度上）扩展大小
    """
    if len(which.shape) == 1:  # 一维直接hstack即可，不区分row或者col，且无法fill diag
        return np.hstack((which, np.zeros(1)))
    if 'row' in extend_what:
        which = np.vstack((which, np.zeros((extend_size, which.shape[1]))))  # 扩展行维度大小
    if 'col' in extend_what:
        which = np.hstack((which, np.zeros((which.shape[0], extend_size))))  # 扩展列维度
    if fill_diag_by != 0 and 'row' in extend_what and 'col' in extend_what:
        # 将对角线fill成1
        for i in range(extend_size):
            which[-i, -i] = fill_diag_by
    return which


class DyLPA:
    def __init__(self, num_nodes, max_iter, min_iter=1, min_loss=1e-3, del_gate=3e-2, decay_alpha=0.09,
                 largest_com=99999, directed=False, scale_estimated=10000, min_at=0.4):
        """
        Constructor
        :param num_nodes:       number of nodes
        :param max_iter: max iteration times of inner loops
        :param min_iter: min iteration times of inner loops
        :param min_loss: min gate of loss. Will break the inner loop if current loss is less than `min_loss`
        :param del_gate:
        :param decay_alpha: weight decay
        :param largest_com:
        :param directed: true if the network is directed
        :param scale_estimated: an estimation number of nodes in the network
        :param min_at: 在多少开始变成min_iter，默认0.4
        """
        self.num_nodes = num_nodes
        self.G = nx.Graph()
        # A是手动维护的adj matrix，暂时只有用来判断是否==0，没有用来计算东西，应该可以用D矩阵代替
        self.A = np.zeros((num_nodes, num_nodes))
        self.D = np.zeros((num_nodes, num_nodes))  # 权重矩阵
        self.D_sum = np.zeros(num_nodes)
        # self.D = np.diag([1.] * num_nodes)
        # self.D_sum = np.ones(num_nodes)
        self.P = np.mat(np.zeros((num_nodes, num_nodes)))  # 转移概率
        # initialize matrix F. Each node is in an independent community at first
        F = np.diag([1.] * num_nodes)  # 初始化partition
        self.F = np.matrix(F)  # matrix 直接用*相乘，np array需要用dot()
        self.Deg = np.zeros(num_nodes)
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.min_loss = min_loss
        self.k = max_iter / (min_at * scale_estimated)
        self.del_gate = del_gate
        self.use_decay = decay_alpha != 0
        self.alpha = decay_alpha
        if self.use_decay:
            global E_REVERSE
            E_REVERSE = 1 / (np.e ** self.alpha)
            print('Weight decay alpha =', self.alpha)
            print('E_REVERSE =', E_REVERSE)
        self.largest = largest_com
        self.mode = 'evolving'
        self.is_directed = directed
        self.num_added_edges = 0
        self.pre_partition = {}

    def increment_mode(self):
        """
        更改为增量模式
        :return:
        """
        self.mode = 'increment'

    def propagation_step(self, node):
        """
        label propagation step
        :param node: a node will propagate its label
        :return: propagated result
        """
        return self.P[node].dot(self.F)

    def append_pre_annotation(self, pre_annotation):
        fixed_partition = {}
        if pre_annotation is not None:
            # if use semi-supervised mode
            # calculate partition
            F_dim1_shape = self.F.shape[1]
            template = np.zeros(F_dim1_shape)  # [0, 0, 0, ...]
            for c in pre_annotation:
                fixed_cid = min(c)
                fixed_partition_for_node = template.copy()  # [0, 0, 0, ...]
                fixed_partition_for_node[fixed_cid] = 1.  # [0, 0, ..., 1.0, 0, ..]
                for node in c:  # 所有c列表中的节点都要被替换成这个partition
                    self.F[node] = fixed_partition[node] = fixed_partition_for_node.copy()
                    self.pre_partition[node] = fixed_cid
        return fixed_partition

    def add_edges(self,
                  edges: list,
                  weights: list,
                  check_interval=0.05,
                  timestamps: Union[list, None] = None,
                  interaction_type: Union[list, None] = None,
                  pre_annotation: list = None):
        """
        Main function of DyLPA
        :param edges: edges formatted in a list of (node1, node2)
        :param weights: weights of each edge. `len(weights)` should equal to `len(edges)`
        :param check_interval: 检查社区，节点合并的间隔
        :param timestamps:  a list of timestamp. `len(timestamps)` should equal to `len(edges)`
        :param interaction_type:    a list of interaction type
        :param pre_annotation:
        :return:
        """

        # if `check_interval` is set to `0`, then check and delete operation will NOT be applied
        # if interaction type is None, initialize it with all "+"
        CHANGE_POINT_INTEGRATION = sys.modules.get('CHANGE_POINT_INTEGRATION', False)
        if CHANGE_POINT_INTEGRATION:
            full_F_records = []
            full_D_records = []

        if interaction_type is None:
            interaction_type = ['+'] * len(weights)

        # ---------- Load pre-annotation ----------
        fixed_partition = self.append_pre_annotation(pre_annotation)

        num_edges = len(edges)
        should_check_del = (check_interval != 1)
        if check_interval != 0:
            check_interval = max(int(num_edges * check_interval), 1)
        t = tqdm(range(num_edges))
        loss_list = []
        pred_result_list = []
        last = None
        if timestamps is not None:
            last = timestamps[0]
        for i in t:
            node1, node2 = edges[i]
            weight = weights[i]

            # decay every propagation, or when time chagnes
            if timestamps is not None:
                self.add_edge_update(node1, node2, weight, last != timestamps[i], interaction_type=interaction_type[i])
                last = timestamps[i]
            else:
                self.add_edge_update(node1, node2, weight, True, interaction_type=interaction_type[i])

            # full_D_records.append(self.D.copy())

            # ------- 添加了一条边 -------
            curr_num_iter = max(-self.k * self.num_added_edges + self.max_iter, 1)
            loss = self._inner_loop(node1, node2, curr_num_iter, fixed_partition)

            loss_list.append(loss)
            if check_interval != 0 and (i + 1) % check_interval == 0:  # 删除社区并合并 todo 动态interval，设计函数实现
                if should_check_del:
                    source_list, by_list, to_del = self.check_del_communities()
                    self.del_and_copy(source_list, by_list, [])  # 现阶段移除删除社区的操作
                    softmax(self.F)
                # 删除以后记录一次预测结果
                curr_prediction = self.predict()
                pred_result_list.append(curr_prediction)  # 只记录partition，不记录communities
            t.set_postfix(loss=format(loss, '.3f'))
            self.num_added_edges += 1  # 已加到图中的边数量+1
            if CHANGE_POINT_INTEGRATION:
                # full_F_records.append(csr_matrix(self.F))
                # full_D_records.append(csr_matrix(self.D))
                full_F_records.append(self.F.copy())
                full_D_records.append(self.D.copy())

        if CHANGE_POINT_INTEGRATION:
            return loss_list, pred_result_list, full_F_records, full_D_records
        else:
            return loss_list, pred_result_list

    def _inner_loop(self, node1, node2, curr_num_iter, fixed_partition=None):
        """
        loop for local propagation
        :param node1: the first node of the edge
        :param node2: the second node of the edge
        :param curr_num_iter: number of iteration in this loop
        :return:    locally propagate results
        """
        if fixed_partition is None:  # better not None
            fixed_partition = {}

        last_node1_community = -1
        last_node2_community = -1
        repeat_times = 0
        loss = 0

        for j in range(min(int(curr_num_iter), self.max_iter)):
            F_i = self.propagation_step(node1)
            if (F_i == 0).all():
                F_i[0, node1] = 1.
            F_j = self.propagation_step(node2)
            if (F_j == 0).all():
                F_j[0, node2] = 1.
            # 计算变化
            curr_loss = (np.abs(F_i - self.F[node1]) + np.abs(F_j - self.F[node2])).sum()  # 只需部分计算
            # 如果有部分标注，则这些的社区分配不可修改
            if node1 in fixed_partition:
                F_i = fixed_partition[node1]
            if node2 in fixed_partition:
                F_j = fixed_partition[node2]
            self.F[node1], self.F[node2] = F_i, F_j  # 不再用copy()，减少复制
            loss += curr_loss

            # 下面的代码段用于处理反复横跳的情况，可以作为删除社区的前提条件之一
            # save current community partition
            curr_node1_community = np.argmax(self.F[node1])
            curr_node2_community = np.argmax(self.F[node2])
            if curr_node1_community == last_node2_community and curr_node2_community == last_node1_community:
                repeat_times += 1
            last_node1_community, last_node2_community = curr_node1_community, curr_node2_community
            if repeat_times >= 5:  # 数字如何确定？
                # 达到阈值开始删除社区
                # 需要格外注意这里使用的索引是否正确（到底是社区索引curr_node1_community，还是节点索引node1）
                if len(self.F[:, curr_node2_community].nonzero()[0]) <= 1:  # 将node2的社区用node1的替换掉
                    self.del_and_copy([node2], [node1], [])
                    softmax(self.F)
                elif len(self.F[:, curr_node1_community].nonzero()[0]) <= 1:  # 与上面的操作反过来
                    self.del_and_copy([node1], [node2], [])
                    softmax(self.F)
            if curr_loss <= self.min_loss:
                break

        # 在这里保存一下Fi和Fj
        # F_records.append([(node1, self.F[node1].copy()), (node2, self.F[node2].copy())])

        return loss

    def add_edge_update(self, node1, node2, weight, decay_on_this_step=True, interaction_type='+'):
        """
        After adding an edge to the network, some matrices should be updated.
        This process is in this function
        :param node1:   the first node in the edge
        :param node2:   the second node in the edge
        :param weight:  the weight of this edge
        :param decay_on_this_step:  if `self.use_decay` is enabled,
                                    and `decay_on_this_step` == true, all edges will decay some value
                                    following the decaying function
        :param interaction_type:    the type of this edge, should be '+' or '-',
                                    meaning adding an edge or remove an edge, respectively
        :return:
        """
        if node1 == node2:  # There should not be a self loop
            return
        # 统计增量方法需要变化的大小，至多在两个维度上各增加2的大小
        row_size = self.F.shape[0]
        n1_not_exists = node1 >= row_size
        n2_not_exists = node2 >= row_size
        increase_size = n1_not_exists + n2_not_exists
        # ---------------- 扩容 ----------------
        if increase_size:
            # 增量演化的尝试
            # 进行扩容
            self.F = extend(self.F, increase_size, ROW_AND_COL, fill_diag_by=1)
            self.D = extend(self.D, increase_size, ROW_AND_COL, fill_diag_by=1)
            self.Deg = extend(self.Deg, increase_size, ROW)
            self.D_sum = extend(self.D_sum, increase_size, ROW)
            self.P = extend(self.P, increase_size, ROW_AND_COL, fill_diag_by=1)
            self.A = extend(self.A, increase_size, ROW_AND_COL, fill_diag_by=1)
        # ---------------- 完成扩容 ----------------

        # change the weight in graph
        if self.G.has_edge(node1, node2):  # not the first time to add this edge
            self.G[node1][node2]['weight'] += weight  # add weight instead of adding a same edge
        else:
            self.G.add_edge(node1, node2, weight=weight)  # first time to add this edge

        # weight decaying（只对D矩阵进行操作）
        if self.use_decay and decay_on_this_step:
            cal_idx = self.D != 0
            new_weight = self.D[cal_idx] * E_REVERSE - E_REVERSE + 1
            self.D[cal_idx] = new_weight

        # ---------------- Update matrices ----------------
        i, j = node1, node2
        if interaction_type == '+':
            self.Deg[i] += weight
            self.Deg[j] += weight
            self.A[i, j] = 1.
            self.D[i, j] += weight  # add weight instead of assigning weight
            self.D_sum[i] += weight
            # undirected
            if not self.is_directed:
                self.A[j, i] = 1.
                self.D[j, i] += weight  # add weight instead of assigning weight
                self.D_sum[j] += weight
        else:
            # update node degree, minus the weight of the disappearing edge
            self.Deg[i] -= self.D[i, j]
            self.A[i, j] = 0  # 邻接置0
            self.D_sum[i] -= self.D[i, j]
            self.D[i, j] = 0
            # 无向图
            if not self.is_directed:
                self.Deg[j] -= self.D[j, i]
                self.A[j, i] = 0
                self.D_sum[j] -= self.D[j, i]
                self.D[j, i] = 0

        # update prob trans matrix
        if self.D_sum[i] != 0:
            self.P[i] = self.D[i] / self.D_sum[i]
        else:
            self.P[i] = 0  # 节点i没有任何边相连，他将不能传播信息给任何其他节点
        if (self.D_sum[j] != 0) and (not self.is_directed):  # 无向图，节点j的转移概率也要更新
            self.P[j] = self.D[j] / self.D_sum[j]
        else:
            self.P[j] = 0  # 节点j没有任何边连接，也要置0

    def check_del_communities(self):
        """
        检查待删除的每一列:
            1. 如果全为0，可以直接删除（因为相当于每一行我只删除了一个0，不会有某一行因为我的操作而变成全0）
            2. 如果有多个不为0的元素，则进一步的检查每一个非0元的所在行是否只有一个元素不为0，如果是，则删除后需要复制
            复制策略: 利用`D`，找到与该行对应节点以最大权重的边连接的点，将该点的社区分配复制过去
            最后返回F
        """
        print('\nChecking no use communities...')
        added_nodes = set(self.G.nodes)  # 获得已添加的节点
        min_nonzero = len(added_nodes) * 0.8
        source_list = []
        by_list = []
        to_del = []  # 待删除的社区列表
        cant_del = [0]
        # check_F = F
        for col_num in range(self.F.shape[1]):  # 列数量是变化的，所以每次要用shape
            if col_num in cant_del:
                continue
            col = self.F[:, col_num]  # 当前检查的列
            col_nonzero_elements = col.nonzero()[0]  # nonzero后是个列向量，所以第1个维度全是0，取[0]维度即可
            col_nonzero_length = len(col_nonzero_elements)
            # 如果全为0，或者（仅有一个元素不为0且该非0元素所在行的那个节点已经添加）
            if col_nonzero_length == 0:
                to_del.append(col_num)
            # 查看是否每个元素都小于一个阈值
            elif (col < self.del_gate).all() \
                    or (col_nonzero_length < min_nonzero and (len(set(col_nonzero_elements) - added_nodes) == 0)):
                to_del.append(col_num)  # 比如(0, 1)边会导致删除两遍，最终没有任何社区了 --- done已修复
                # 希望`is_row_unique_zero`能够做到这么一件事的判别：
                # 如果某次的F长这样：[[0.5, 0.5, 0, 0, 0],
                #                  [0.5, 0.5, 0, 0, 0],
                #                  [0,   0,   0, 0, 0],
                #                  [0,   0,   0, 0, 0],
                #                  [0,   0,   0, 0, 0]]
                # 且此时col_num=0，则is_row_unique_nonzero=True，表示并非第0列的所有元素都是它所在行唯一的非0元
                # 如果某次的F长这样：[[0.5, 0, 0, 0, 0],
                #                  [0.5, 0, 0, 0, 0],
                #                  [0,   0, 0, 0, 0],
                #                  [0,   0, 0, 0, 0],
                #                  [0,   0, 0, 0, 0]]
                # 此时若col_num=0，那么就要求is_row_unique_nonzero=False，表示这一列不能删除，删除了会导致全0元且无法用复制弥补
                is_row_unique_nonzero = True
                all_not_max = True
                for target_row in col_nonzero_elements:
                    # 按行取，元组中第一个元素若非空则必全都是0
                    row_nonzero_elements = self.F[target_row].nonzero()[1]  # 直接取1号元素：ndarray([x, y, z])
                    row_max_element = np.argmax(self.F[target_row])
                    if row_max_element == col_num:
                        all_not_max = False
                    if len(row_nonzero_elements) > 1:
                        is_row_unique_nonzero = False
                    max_weight_node = self.find_copy_target(target_row)
                    cant = self.F[max_weight_node].nonzero()[1].tolist()
                    if not is_row_unique_nonzero:
                        if col_num in cant and all_not_max:
                            # 如果当前正在检查的就是当前列，则它不受cant的限制，只受can的限制
                            cant.remove(col_num)
                    cant_del.extend(cant)
                    if len(row_nonzero_elements) == 1:  # 这一行只有一个非0元素，需要复制
                        # 注：这里不需要判断该非零元是不是正好是col_num，因为这里的行都是从col_nonzero_elements里面得到的
                        # 通过D矩阵找到该节点与哪个节点的权重最大，找到那节点并复制概率
                        if len(cant) > 1:
                            source_list.append(target_row)
                            by_list.append(max_weight_node)
        to_del = list(set(to_del) - set(cant_del))
        return source_list, by_list, to_del

    def find_copy_target(self, node_id, method='max weight'):
        """
        Find a node whose community partition can best replace the partition of `node_id`
        Now maximize the modularity to choose the target node_id.
        ----------
        Tried:
            - Max weight node
            - Maximize modularity
        :param node_id:
        :param method:
        :return:
        """
        if method == 'max weight':
            max_weight_node = np.argmax(self.D[node_id])
            return max_weight_node
        elif method == 'max modularity':
            p, _ = self.predict()
            original = p[node_id]
            modularity_list = []
            for i in range(self.D.shape[0]):
                if i == node_id:
                    p[node_id] = original
                else:
                    p[node_id] = p[i]
                modularity_list.append(modularity(p, self.G))
            # get argmax
            argmax = np.argmax(modularity_list)
            return argmax

    def del_and_copy(self, source_list, by_list, to_del_list):
        """
        用by替换掉source；再删除to_del
        """
        # 先复制再删除
        if len(source_list) != 0:
            self.copy_community(source_list, by_list)
        if len(to_del_list) != 0:
            self.del_community(to_del_list)

    def copy_community(self, source, by):
        """
        用`by`的社区分配替换掉``source`的社区分配
        NOTE: 直接在原始的F上进行的修改，没有返回值
        source: 列表或int
        by: 列表或int
        """
        self.F[source] = self.F[by]

    def del_community(self, to_del):
        """
        删除`to_del`的社区并返回删除后的社区
        """
        self.F = np.delete(self.F, to_del, axis=1)  # 删除社区（删除列）

    def predict(self, F=None, G=None, level: Union[int, None] = 3):
        """

        :param F:
        :param G:
        :param level:
        :return:
        """
        # get community assignment
        if F is None or G is None:
            F, G = self.F, self.G

        # 构造预标注矩阵`pre_partition_converted`，之后就可以
        # 直接方便的F[annotated_nodes] = pre_partition_converted
        annotated_nodes = list(self.pre_partition.keys())
        pre_partition_converted = np.zeros((len(annotated_nodes), self.F.shape[1]))
        for i, nid in enumerate(annotated_nodes):
            pre_partition_converted[i] = self.pre_partition[nid]

        for _ in range(level):
            F = np.dot(self.P, F)
            if len(annotated_nodes) > 0:  # 只有存在预标注的时候才执行这个操作
                # 将标注的节点恢复
                F[annotated_nodes] = pre_partition_converted

        assignment = np.argmax(np.array(F), axis=1)
        partition = {}
        for i in range(F.shape[0]):
            partition[i] = assignment[i]
        communities = partition2community(partition)
        return partition, communities
