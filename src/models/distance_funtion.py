# %%
import math
import pickle
from abc import ABCMeta, abstractmethod
from statistics import mean, median
from typing import Union

import numpy as np
import tqdm
from matplotlib import pyplot as plt


class DistanceComputable:
    __metaclass__ = ABCMeta

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, mat1: Union[np.ndarray, np.matrix], mat2: Union[np.ndarray, np.matrix]):
        raise NotImplementedError('`compute` is not implemented')


class NormComputable(DistanceComputable):
    @abstractmethod
    def compute(self, mat: Union[np.ndarray, np.matrix], *args):
        raise NotImplementedError('`compute` is not implemented')


class DistanceFunction:
    """
    距离函数
    """

    def __init__(self, dataset: str, distance_computable: DistanceComputable):
        super().__init__()
        self.dataset = dataset
        self.name = distance_computable.name
        self._compute = distance_computable.compute
        self.stored_result = None
        self.stored_change_rate_result = None
        self.last_result = None

    def compute_and_store(self, matrix_list: Union[list, np.ndarray], is_increment: bool = False) -> None:
        """
        根据传入的矩阵列表计算矩阵的距离并存储到内部
        :param matrix_list:     待计算的矩阵列表
        :param is_increment:    是否增量计算，如果True，那么`self.stored_result`不会被清空，会扩增
        :return:    None
        """
        if self.last_result is not None and is_increment:
            # 只有这种情况下才需要用增量计算方式
            temp_result = [self._compute(self.last_result, matrix_list[0])]
            for i in tqdm.tqdm(range(1, len(matrix_list))):
                last, curr = matrix_list[i - 1], matrix_list[i]
                temp_result.append(self._compute(last, curr))
            self.stored_result.extend(temp_result)
            # 计算变化率，也涉及到是否增量（为了减少计算量）
            # 这里需要用extend
            self.stored_change_rate_result.extend(self.compute_change_rate(temp_result, is_increment=is_increment))
        else:
            # 没有记录过先前的最后一个矩阵，无论是否increment，全部直接重新计算
            self.stored_result = [0]
            for i in tqdm.tqdm(range(1, len(matrix_list))):
                last, curr = matrix_list[i - 1], matrix_list[i]
                self.stored_result.append(self._compute(last, curr))
            # 这里直接进行赋值即可
            self.stored_change_rate_result = self.compute_change_rate()
        self.last_result = matrix_list[-1]

    def check_has_stored_result(self):
        return self.stored_result is not None

    def check_stored_result_and_raise_error(self):
        if not self.check_has_stored_result():
            raise ValueError('You should call `compute_and_store()` first.')

    def draw(self, left=0, right=-1):
        if not self.check_has_stored_result():
            raise ValueError('You should call `compute_and_store()` first.')
        plt.plot(self.stored_result[left: right])
        name_temp = self.name if self.name.lower().endswith(' distance') else self.name + ' distance'
        plt.title('Value of {} through time on \n data {}'.format(name_temp, self.dataset))
        plt.show()

    def compute_change_rate(self, difference_list: list = None, is_increment=False):
        """
        根据传入的`difference_list`或者`self_stored_result`计算元素之间的变化率
        只有当`is_increment`为True时，才会使用`difference_list`进行计算，
            否则会直接使用`self.stored_result`
        :param difference_list:
        :param is_increment:
        :return:
        """
        if not self.check_has_stored_result() and difference_list is None:
            raise ValueError('You should call `compute_and_store()` first or pass in `difference_list`')
        if is_increment:
            cr_list_incremental = [
                np.abs((difference_list[0] - self.stored_result[-1]) / self.stored_result[-1])]  # cr: change rate
            last = difference_list[0]
            for i in range(1, len(difference_list)):
                distance = difference_list[i]
                cr_list_incremental.append(np.abs((distance - last) / last))
                last = distance
            return cr_list_incremental
        else:
            last = self.stored_result[0]
            cr_list = [0]  # cr: change rate
            for i in range(1, len(self.stored_result)):
                distance = self.stored_result[i]
                cr_list.append(np.abs((distance - last) / last))
                last = distance
            return cr_list

    def draw_change_rate(self):
        cr_list = self.compute_change_rate()
        plt.plot(cr_list)
        name_temp = self.name if self.name.lower().endswith(' distance') else self.name + ' distance'
        plt.title('Change rate of {} through time on data \n{}'.format(name_temp, self.dataset))
        plt.show()
        removed_nan = [i for i in cr_list if i != np.nan]
        print('{} info: \nmean={}\nmedian={}\nmax={}\nmin={}'.format(name_temp, mean(removed_nan),
                                                                     median(removed_nan),
                                                                     max(removed_nan),
                                                                     min(removed_nan)))

    def save_computed_results(self, appendix: str = ''):
        if not self.check_has_stored_result():
            raise ValueError('You should call `compute_and_store()` first.')
        f = open('/home/zhuyeqi/changepoint/cpd_temp_results/{}_{}_{}.pkl'.format(self.dataset,
                                                                                  self.name,
                                                                                  appendix), 'wb')
        pickle.dump(self.stored_result, f)
        f.close()

    def load_save_computed_results(self, appendix: str = ''):
        f = open('/home/zhuyeqi/changepoint/cpd_temp_results/{}_{}_{}.pkl'.format(self.dataset,
                                                                                  self.name,
                                                                                  appendix), 'rb')
        self.stored_result = pickle.load(f)
        f.close()
        # compute change rate
        self.stored_change_rate_result = self.compute_change_rate()

    def predict(self, threshold):
        if not self.check_has_stored_result():
            raise ValueError('You should call `compute_and_store()` first.')
        change_point_list = []
        for i, d1cr in enumerate(self.stored_change_rate_result):
            if d1cr >= threshold and d1cr != math.inf:
                change_point_list.append(i)
        return change_point_list

    def top_k_predict(self, k: int):
        """
        top k select change point
        :param k:
        :return:
        """
        self.check_stored_result_and_raise_error()
        copied = self.stored_change_rate_result.copy()
        # filter nan and inf
        for i in range(len(copied)):
            if np.isinf(copied[i]) or np.isnan(copied[i]):
                copied[i] = 0
        cp_list = np.argsort(copied, )[::-1][:k]
        print('当前等效阈值为{}'.format(copied[cp_list[-1]]))
        return np.sort(cp_list)


class NormDistance(DistanceFunction):
    # 只需要把这个覆盖掉即可实现"先计算norm，再计算距离"
    # 而此前的是"先计算delta，再计算norm"（从外部看其实就是矩阵距离）

    def __init__(self, dataset: str, distance_computable: NormComputable):
        super().__init__(dataset, distance_computable)
        self._compute = distance_computable.compute

    def compute_and_store(self, matrix_list: Union[list, np.ndarray], is_increment: bool = False) -> None:
        """
        根据传入的矩阵列表计算矩阵的距离并存储到内部
        :param matrix_list:     待计算的矩阵列表
        :param is_increment:    是否增量计算，如果True，那么`self.stored_result`不会被清空，会扩增
        :return:    None
        """
        if not is_increment or self.stored_result is None:
            self.stored_result = []
        temp_result = []

        for i in tqdm.tqdm(range(len(matrix_list))):
            temp_result.append(self._compute(matrix_list[i], ))
        self.stored_result.extend(temp_result)
        if is_increment:
            # is increment, use extend
            if self.stored_change_rate_result is None:
                self.stored_change_rate_result = []
            self.stored_change_rate_result.extend(self.compute_change_rate(temp_result, is_increment=is_increment))
        else:
            # not increment
            self.stored_change_rate_result = self.compute_change_rate()
