import os
import pickle
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any

import src.metrics.cpd_metrics as m


class ChangePointDetectionAlgorithmComponent:
    __metaclass = ABCMeta

    def __init__(self, exp_dir_prefix=None, data_dir_prefix=None) -> None:
        if exp_dir_prefix is None:
            # self.exp_dir_prefix = '/data/share_data/DyLPA_experiment_temp/experiments/'
            self.exp_dir_prefix = '/home/zhuyeqi/DyLPA_experiment_temp/experiments/'
        else:
            self.exp_dir_prefix = exp_dir_prefix
        if data_dir_prefix is None:
            self.data_dir_prefix = 'data/'
        else:
            self.data_dir_prefix = data_dir_prefix

    @property
    @abstractmethod
    def approach_name(self) -> str:
        raise NotImplementedError('Not implemented')

    def get_exp_save_dir(self, data_name) -> str:
        return os.path.join(self.exp_dir_prefix, data_name, self.approach_name)

    def get_data_dir(self, data_name):
        return os.path.join(self.data_dir_prefix, data_name)

    def get_data_path(self, data_name):
        return os.path.join(self.data_dir_prefix, data_name, 'data.pkl')

    # @abstractmethod
    # def _load_data(self, data_name, dataloader, **kwargs):
    #     raise NotImplementedError('`load_data` is not implemented')

    def execute(self, data_name: str, data: Any, including_eval=False, **kwargs):
        """
        execute train and evaluation
        if the return result of `train` follows the data type (prediction: Any, time_cost: float),
        then it's no need to override the `execute` function.
        If not, you may need to change the return content of `train`, or override this function.
        :param data_name:
        :param data:
        :param including_eval:  if true, then the evaluation will be executed as well.
                                In this case, `target` must be in `kwargs`
        :param kwargs:  Any parameters that the `train` and `evaluate` may need
        :return:
        """
        pred, time_cost = self.train(data, data_name=data_name, **kwargs)

        result = {
            'prediction': pred,
            'time_cost': time_cost,
        }

        if including_eval:
            assert 'target' in kwargs, 'if evaluation is included, the `target` should be passed in.'
            f1, p, r, delta_t = self.evaluate(prediction=pred, target=kwargs['target'])
            result['evaluation'] = {
                'F1': f1,
                'P': p,
                'R': r,
                'delta_t': delta_t,
            }

        return result

    def dump(self, exp_result, data_name, **kwargs) -> None:
        """
        dump `exp_result`
        :param exp_result:
        :param data_name:
        :param kwargs:
        :return:
        """
        exp_dir = self.get_exp_save_dir(data_name)
        suffix = kwargs.get('suffix', '')
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, 'cpd_result{}.pkl'.format(suffix)), 'wb') as f:
            pickle.dump(exp_result, f)

    def load(self, data_name, **kwargs) -> Any:
        """
        load experiments
        :param data_name:
        :param kwargs:
        :return:
        """
        suffix = kwargs.get('suffix', '')
        exp_path = os.path.join(self.get_exp_save_dir(data_name), 'cpd_result{}.pkl'.format(suffix))
        if not os.path.exists(exp_path):
            warnings.warn('You may have not dumped the experiment result.')
            return None
        with open(exp_path, 'rb') as f:
            return pickle.load(f)

    @abstractmethod
    def train(self, data, **kwargs):
        raise NotImplementedError('`train` is not implemented')

    def evaluate(self, metrics=None, **kwargs):
        prediction = kwargs['prediction']
        target = kwargs['target']
        margin = kwargs.get('margin', 5)

        f1, p, r = m.cpd_f1_with_margin(target, prediction, margin, return_PR=True)
        delta_t = m.delta_t(prediction, target)
        # delta_t = 'Currently Not Available'

        return f1, p, r, delta_t
