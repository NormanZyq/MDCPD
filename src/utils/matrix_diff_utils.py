import sys
from typing import Union, List

import numpy as np

from src.models.distance_funtion import DistanceFunction


def adjust_F_list(F_list: np.ndarray, ground_truth_list: np.ndarray, start: int = 1000, end: int = -1, plus: int = 1):
    for idx in range(start, end):
        ground_truth = ground_truth_list[idx]
        for node in ground_truth:
            c = ground_truth[node]
            F_list[idx][node][c] += plus
        # standardization
        mu = np.mean(F_list[idx], axis=1)
        sigma = np.std(F_list[idx], axis=1)

        std_f = ((F_list[idx].T - mu) / sigma).T
        # normalization
        min_f = np.min(std_f)
        F_list[idx] = (std_f - min_f) / (np.max(std_f) - min_f)


def adjust_F(F, ground_truth, intervene_nodes, plus=1):
    for node in intervene_nodes:
        c = ground_truth[node]
        F[node][c] += plus
        # standardization
        mu = np.mean(F[node], )
        sigma = np.std(F[node], )

        std_f = (F[node] - mu) / sigma
        # normalization
        min_f = np.min(std_f)
        F[node] = (std_f - min_f) / (np.max(std_f) - min_f)
    return F


def delta_t_eval(prediction, target) -> int:  # ONlogM -- M: len(pred)
    """
    该函数满足以下需求：
    遍历target的每个元素i，从prediction中找到与i的差值的绝对值最小的元素，设为k，
    找出所有的k，让它与对应的i作差取绝对值，最后求和并返回
    :param prediction: 预测演化点的t序列
    :param target: 真实演化点所在t的序列
    :return: 参见上文描述
    """
    delta_t_sum = 0
    result = []
    for a in target:
        min_diff = abs(a - prediction[0])
        b = prediction[0]
        for p in prediction:
            diff = abs(a - p)
            if diff < min_diff:
                min_diff = diff
                b = p
        result.append(b)
    for i in range(len(result)):
        delta_t_sum += abs(result[i] - target[i])

    return delta_t_sum


def print_distance(left: int, right: int, compare_obj: Union[DistanceFunction, List[DistanceFunction]]):
    """
    以字符形式打印距离以供直观比较

    e.g.,
    print_distance(855, 865, [d1, d2, d3, abs_d3, manhattan, chebyshev])

    :param left: 下界
    :param right: 上届（包含）
    :param compare_obj: 需要比较的东西
    :return: None
    """
    # convert to list
    if type(compare_obj) == DistanceFunction:
        compare_obj = [compare_obj]
    # check value legality
    for func in compare_obj:
        if not func.check_has_stored_result():
            raise ValueError(
                '{} does not have stored results. You should call `compute_and_store()` first'.format(func.name))

    idx_list = range(left, right + 1)
    print('Distance')
    for idx in idx_list:
        output_str = 'idx=' + str(idx)
        for func in compare_obj:
            output_str += '\t'
            output_str += func.name + '=' + format(func.stored_result[idx], '.7f')
        print(output_str)


def print_change_rate(left: int, right: int, compare_obj: Union[DistanceFunction, List[DistanceFunction]]):
    """
    类似`print_distance`，但打印的是chage rate
    :param left:
    :param right:
    :param compare_obj:
    :return:
    """

    # convert to list
    if type(compare_obj) == DistanceFunction:
        compare_obj = [compare_obj]
    # check value legality
    for func in compare_obj:
        if not func.check_has_stored_result():
            raise ValueError(
                '{} does not have stored results. You should call `compute_and_store()` first'.format(func.name))

    # 计算change rate
    change_rate_lists = {
        obj.name: obj.compute_change_rate() for obj in compare_obj
    }

    idx_list = range(left, right + 1)
    print('Change rate')
    for idx in idx_list:
        output_str = 'idx=' + str(idx)
        for func in compare_obj:
            output_str += '\t'
            output_str += func.name + '=' + format(change_rate_lists[func.name][idx], '.7f')
        print(output_str)


# %%
def combineA_F(A_list, F_list):
    # 干预F
    for idx in range(len(F_list)):
        f_matrix = F_list[idx]
        a_matrix = A_list[idx]
        for i in range(a_matrix.shape[0]):
            for j in range(a_matrix.shape[1]):
                if a_matrix[i][j] >= 1:
                    f_matrix[i][j] = f_matrix[j][i] = max(f_matrix[i][j], f_matrix[j][i])


def D_intervene_F(D_list, F_list):
    """
    理论分析认为它不需要通过迭代达到收敛。
    这个函数的功能如下：
        通过`D_list`获得与每个节点i的度最大的节点j
        然后通过`F_list`获知j被分配到c社区的概率最大
        然后将节点i被分配到c社区的概率调大，调整到与它原先被分配到的社区一样的概率`max(F_ic)`
    :param D_list:
    :param F_list:
    :return:
    """
    print('Optimizing `F` by `D`')
    mapping = {}
    for idx in range(len(D_list)):
        f, d = F_list[idx], D_list[idx]
        for i in range(d.shape[0]):
            mapping[i] = np.argmax(d[i])
        for i in mapping:
            j = mapping[i]
            c = np.argmax(f[j])
            f[i, c] = f[i].max()
    # 由于直接修改了矩阵，所以不需要返回值


def is_in_margin(p, g, m):
    return abs(p - g) <= m


def precision_eval(prediction, target, all_snapshot_list, margin=0) -> float:
    tp, fp = 0, 0
    prediction = set(prediction)
    target = set(target)
    used_indices = set()
    for time_step in all_snapshot_list:
        if time_step in target:
            # tp_exist = False
            # for p in prediction:
            #     if p in used_indices:    # 如果已使用，就跳过
            #         continue
            #     if time_step - p > margin:      # 减少不必要的遍历
            #         break
            #     if is_in_margin(p, time_step, margin):
            #         tp_exist = True
            #         used_indices.add(p)
            #         break
            # if tp_exist:
            #     tp += 1
            # else:
            #     pass        # fn no need
            if time_step in prediction:
                tp += 1
            else:
                pass  # no need
        else:  # not in target
            # positive_sample_exists = False
            # for p in prediction:
            #     if p in used_indices:  # 如果已使用，就跳过
            #         continue
            #     if time_step - p > margin:  # 减少不必要的遍历
            #         break
            #     if is_in_margin(p, time_step, margin):
            #         positive_sample_exists = True
            #         break
            # if positive_sample_exists:
            #     fp += 1
            if time_step in prediction:
                fp += 1
            else:
                pass  # no need for precision

    return tp / (tp + fp)


def recall_eval(prediction, target, all_snapshot_list, margin=0) -> float:
    tp, fn = 0, 0,
    prediction = set(prediction)
    target = set(target)
    used_indices = set()
    for time_step in all_snapshot_list:
        if time_step in target:
            # tp_exist = False
            # for p in prediction:
            #     if p in used_indices:  # 如果已使用，就跳过
            #         continue
            #     if time_step - p > margin:  # 减少不必要的遍历
            #         break
            #     if is_in_margin(p, time_step, margin):
            #         tp_exist = True
            #         used_indices.add(p)
            #         break
            # if tp_exist:
            #     tp += 1
            # else:
            #     fn += 1
            if time_step in prediction:
                tp += 1
            else:
                fn += 1
        else:
            pass  # recall不需要这一段

    return tp / (tp + fn)


def f1_eval(prediction, target, all_snapshot_list, margin=0):
    p = precision_eval(prediction, target, all_snapshot_list, margin)
    r = recall_eval(prediction, target, all_snapshot_list, margin)
    return 2 * (p * r) / (p + r)


def enable_change_point():
    sys.modules['CHANGE_POINT_INTEGRATION'] = True


def disable_change_point():
    sys.modules['CHANGE_POINT_INTEGRATION'] = False


def f1_eval2(pred, target, all_snapshot_list, M):
    used_indices = set()
    tp, fp, fn = 0, 0, 0

    for t in target:
        tp_exists = False
        for i, p in enumerate(pred):
            if p not in used_indices and abs(p - t) <= M:
                tp_exists = True
                used_indices.add(i)
                break
        if tp_exists:
            tp += 1
    P = tp / len(pred)
    R = tp / len(target)
    return P, R, 2 * (P * R) / (P + R)


def true_positives(T, X, margin=5):
    """Compute true positives without double counting
    >>> true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> true_positives(set(), {1, 2, 3})
    set()
    >>> true_positives({1, 2, 3}, set())
    set()
    """
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """Compute the F-measure based on human annotations.
    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too
    Remember that all CP locations are 0-based!
    >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> f_measure({1: [], 2: [10], 3: [50]}, [])
    0.8
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F
