from typing import Union

import numpy as np

from src.models.distance_funtion import NormComputable


# 定义各个距离的计算函数
class L1Norm(NormComputable):
    def __init__(self):
        super().__init__('L1Norm')

    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        return np.linalg.norm(mat, ord=1)


class L2Norm(NormComputable):
    def __init__(self):
        super().__init__('L2Norm')

    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        return np.linalg.norm(mat, ord=2)


class FroNorm(NormComputable):
    def __init__(self):
        super().__init__('FrobeniusNorm')

    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        return np.linalg.norm(mat, ord='fro')


class InfiniteNorm(NormComputable):
    def __init__(self):
        super().__init__('d1')

    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        return np.linalg.norm(mat, ord=np.inf)


class SingularValueSum(NormComputable):
    """
    求top k个奇异值并求和
    初步实验结论：当k较小时性能尚可，k=1时就是L2 Norm，目前性能最好
    """
    def __init__(self, num_singular):
        super().__init__('SingularValueSum')
        self.num_singular = num_singular

    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        num = self.num_singular
        _, s, _ = np.linalg.svd(mat)
        if len(s) < num:
            print('in')
            num = len(s)
        return np.sum(s[:num])


class SingularValueMean(NormComputable):
    """
    取top k 个奇异值并求均值，如果某个矩阵>0的的奇异值数量小于k，则只对那几个求平均
    初步实验结论：表现类似于sum版，F矩阵似乎没有等于0的奇异值
    """
    def __init__(self, num_singular):
        super().__init__('SingularValueMean')
        self.num_singular = num_singular

    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        num = self.num_singular
        _, s, _ = np.linalg.svd(mat)
        return np.mean(s[:num])


class SingularValuePaddingMean(NormComputable):
    """
    取top k个奇异值，求均值，这k个中允许包含0，如果不足k个>0的，则补齐到k
    初步实验结论：不可这么做，效果不好
    """
    def __init__(self, num_singular):
        super().__init__('SingularValueMean')
        self.num_singular = num_singular

    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        num = self.num_singular
        _, s, _ = np.linalg.svd(mat)
        if len(s.nonzero()[0]) < len(s):
            print(len(s.nonzero()[0]))

        for i in range(len(s), num):
            print('in')

            s.append(0)
        return np.mean(s)
