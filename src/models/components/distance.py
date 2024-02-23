from typing import Union

import numpy as np

from src.models.distance_funtion import DistanceComputable


# 定义各个距离的计算函数
class D1(DistanceComputable):
    def __init__(self):
        super().__init__('d1')

    def compute(self, mat1: Union[np.ndarray, np.matrix], mat2: Union[np.ndarray, np.matrix]):
        abs_minus_result = np.abs(mat1 - mat2)
        return np.sum(abs_minus_result)


class ManhattanDistance(DistanceComputable):
    def __init__(self):
        super().__init__('Manhattan')

    def compute(self, mat1: Union[np.ndarray, np.matrix], mat2: Union[np.ndarray, np.matrix]):
        return np.linalg.norm(mat1 - mat2, ord=1)


class D3(DistanceComputable):
    def __init__(self):
        super().__init__('d3')

    def compute(self, mat1: Union[np.ndarray, np.matrix], mat2: Union[np.ndarray, np.matrix]):
        minus_result = mat1 - mat2
        row_max = np.max(minus_result, axis=1)
        return np.max(row_max)


class AbsD3(DistanceComputable):
    def __init__(self):
        super().__init__('abs(d3)')

    def compute(self, mat1: Union[np.ndarray, np.matrix], mat2: Union[np.ndarray, np.matrix]):
        minus_result = np.abs(mat1 - mat2)
        row_max = np.max(minus_result, axis=1)
        return np.max(row_max)


class D2(DistanceComputable):
    def __init__(self):
        super().__init__('d2')

    def compute(self, mat1: Union[np.ndarray, np.matrix], mat2: Union[np.ndarray, np.matrix]):
        A_B = mat1 - mat2
        return np.sqrt(np.trace(np.dot(A_B, A_B.T)))


class ChebyshevDistance(DistanceComputable):
    def __init__(self):
        super().__init__('Chebyshev')

    def compute(self, mat1: Union[np.ndarray, np.matrix], mat2: Union[np.ndarray, np.matrix]):
        return np.linalg.norm(mat1 - mat2, ord=np.inf)
