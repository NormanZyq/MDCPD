import os.path
import time
from typing import Any

import numpy as np
import scipy.stats

from src.baselines.change_point.cicpd_components.cicpd_base import detect_change_points
from src.baselines.cpd_baseilne import ChangePointDetectionAlgorithmComponent


def kernel_js(p, q):
    '''
    :param p:
    :param q:
    :return:
    '''
    M = (p + q) / 2
    # defaults to ``e``
    js2 = 0.5 * scipy.stats.entropy(p, M, base=np.e) + \
          0.5 * scipy.stats.entropy(q, M, base=np.e)
    return js2


class CICPD(ChangePointDetectionAlgorithmComponent):
    def __init__(self, exp_dir_prefix=None, data_dir_prefix=None) -> None:
        super().__init__(exp_dir_prefix, data_dir_prefix)

    @property
    def approach_name(self) -> str:
        return 'cicpd'

    def execute(self, data_name: str, data: Any, including_eval=False, **kwargs):
        return super().execute(data_name=data_name, data=data, including_eval=including_eval, **kwargs)

    def train(self, data, **kwargs):
        target = kwargs['target']

        alpha_pg = 0.85
        s = 0
        # load save dir and path
        network_path = os.path.join(self.exp_dir_prefix, self.approach_name, kwargs.get('mtx_name', 'tmp.mtx'))
        ground_truth = set(target[1])
        # LeaderRank or PageRank
        nodeImportance = kwargs['node_importance']
        # flag1 1 for True(original) 0 for False(supplement)
        flag1 = kwargs['flag1']
        # flag2 1 for True(value) 0 for False(normalized_rank)
        flag2 = kwargs['flag2']
        # kernel mode: js(js_distance)
        kernel_mode = kwargs['kernel_mode']
        flag_dict = {"0": False, "1": True}
        flag1 = flag_dict[flag1]
        flag2 = flag_dict[flag2]

        print("data", network_path)
        print("nodeImportance", nodeImportance)
        print("flag1,flag2", flag1, flag2)
        print("kernelFun", kernel_mode)
        print("groundTruth", ground_truth)

        # ground_truth = groundTruth_dict[ground_truth]
        print("ground: ", ground_truth)
        print("sliding length: ", s)
        kernel_mode_dict = {"js": kernel_js}
        kernel_fun = kernel_mode_dict[kernel_mode]
        t1 = time.time()
        p_c, _, _, _ = detect_change_points(network_path,
                                            ground_truth,
                                            nodeImportance,
                                            alpha_pg,
                                            flag1,
                                            flag2,
                                            s, kernel_fun)
        t2 = time.time()
        return list(p_c), t2 - t1

    def process_rdyn_data(self, data_name: str, num_nodes: int):
        if data_name.endswith('-snapshot'):
            data_name = data_name[:-9]
        f = open(os.path.join(self.data_dir_prefix, data_name, 'interactions.txt'), 'r')
        # load interactions
        lines = []
        for line in f:
            lines.append(line.split('\t'))
        f.close()
        row_size = col_size = num_nodes  # 根据实际情况

        def get_index(row, col):
            return row * col_size + col

        # 写文件
        # 期望的格式：snap_id \t 横着数的索引号 \t weight
        mat = np.zeros((row_size, col_size))

        # save dir
        save_dir = os.path.join(self.exp_dir_prefix, self.approach_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, 'tmp.mtx')
        # 如果文件存在则删除
        if os.path.exists(filepath):
            os.remove(filepath)
        f = open(filepath, 'w')
        f.write('%%MatrixMarket matrix coordinate real general\n%\n')  # 信息行
        # f.write('500\t250000\t2040\n')  # 首行
        f.write('{}\t{}\t{}\n'.format(num_nodes, row_size * col_size, len(lines)))

        last_snapshot_id = -1
        last_batch = []
        for line in lines:
            curr_snapshot_id = int(line[0])
            node1, node2 = int(line[3]), int(line[4])
            if curr_snapshot_id != last_snapshot_id:
                # next
                # do some operations and write the file
                # now computation is finished, output to the file
                for lb in last_batch:
                    index = get_index(lb[0], lb[1])
                    weight = mat[lb[0], lb[1]]
                    # if weight > 0:
                    f.write('{}\t{}\t{}\n'.format(last_snapshot_id + 1, index, weight))
                last_snapshot_id += 1
                last_batch.clear()
            # same snapshot id, or the first line with the new snapshot id
            if line[2] == '+':
                mat[node1][node2] += 1
            else:
                curr = mat[node1, node2]
                # although sometimes the edge disappears, we should still keep the edge
                #   in the graph. Because the approach does not support negative weight edge or zero
                if curr >= 1:  # only when >= 1 then minus the weight
                    mat[node1][node2] -= 1
            last_batch.append([node1, node2])

        # add the content in the final batch
        for lb in last_batch:
            index = get_index(lb[0], lb[1])
            weight = mat[lb[0], lb[1]]
            # if weight > 0:
            f.write('{}\t{}\t{}\n'.format(last_snapshot_id + 1, index, weight))

        f.close()
