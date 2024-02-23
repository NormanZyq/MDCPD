def _true_positives(T, X, margin=5):
    """Compute true positives without double counting
    >>> _true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> _true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> _true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> _true_positives(set(), {1, 2, 3})
    set()
    >>> _true_positives({1, 2, 3}, set())
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


def cpd_f1_with_margin(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """Compute the F-measure based on human annotations.
    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too.
    Remember that all CP locations are 0-based!
    >>> cpd_f1_with_margin({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> cpd_f1_with_margin({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> cpd_f1_with_margin({1: [], 2: [10], 3: [50]}, [])
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

    P = len(_true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: _true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def delta_t(prediction, target) -> int:  # ONlogM -- M: len(pred)
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
    if type(target) == dict:
        target = target[1]
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
