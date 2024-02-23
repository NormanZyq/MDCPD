"""
Evaluation metrics for Community Detection
"""

import math


def normalized_mutual_information(c1, c2):
    """
    Calculate Normalized Mutual Information (NMI) of community detection result c1 and ground_truth c2.
    Format:
        c1 = [[1, 2, 3], [4, 5, 6]]
        c2 = [[1, 2 ,3], [4, 5, 6]]
        nmi = NMI(c1, c2)
        print(nmi)
    Return: NMI value in range [0, 1]
    Reference: Danon L, Diazguilera A, Arenas A. Effect of size heterogeneity
               on community identification in complex networks[J]. Arxiv Physics, 2006, 2006(11):11010.
    """
    nmi = 0
    u = 0
    d = 0
    n1 = 0
    n2 = 0
    for i in c1:
        n1 += len(i)
    for j in c2:
        n2 += len(j)
    assert n1 == n2, 'ERROR: Size of c1 is not equal to c2'

    s = n1

    for i in c1:
        ni = len(i)
        for j in c2:
            nj = len(j)
            nij = len(set(i) & set(j))
            if nij == 0:
                continue
            logt = math.log((nij * s) / (ni * nj))
            u += nij * logt
    u *= -2

    for i in c1:
        ni = len(i)
        d += ni * math.log(ni / s)
    for j in c2:
        nj = len(j)
        d += nj * math.log(nj / s)

    if d != 0:
        nmi = u / d

    return nmi


def f1_components(prediction, target):
    tp, tn, fp, fn = 0, 0, 0, 0
    for n1 in prediction:
        for n2 in prediction:
            if n1 >= n2:
                continue
            # 数正确与否的对数
            if target[n1] == target[n2]:
                if prediction[n1] == prediction[n2]:
                    tp += 1
                else:
                    fn += 1
            else:
                if prediction[n1] == prediction[n2]:
                    fp += 1
                else:
                    tn += 1
    return tp, tn, fp, fn


def f1_score(prediction=None, target=None, tp=None, tn=None, fp=None, fn=None):
    if tp is not None and tn is not None and fp is not None and fn is not None:
        return 2 * tp / (2 * tp + fn + fp)
    tp, tn, fp, fn = f1_components(prediction, target)
    return 2 * tp / (2 * tp + fn + fp)


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def accuracy(tp, tn, fp, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc


def normalized_f1_score(prediction, target):
    tp, tn, fp, fn = f1_components(prediction, target)
    f1 = f1_score(tp=tp, tn=tn, fp=fp, fn=fn)
    p = precision(tp, fp)
    # r = recall(tp, fn)        # will not be used as metric or to cal F1
    f1_coin = 2 * p / (p + 1)
    f1_norm = (f1 - f1_coin) / (1 - f1_coin)
    return f1_norm
