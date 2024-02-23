import math
import time

import numpy as np

import src.models.components.distance as d
import src.utils.matrix_diff_utils as mdu
from src.baselines.cpd_baseilne import ChangePointDetectionAlgorithmComponent
from src.dy_lpa import DyLPA
from src.models.distance_funtion import DistanceFunction


class MDCPD(ChangePointDetectionAlgorithmComponent):
    def __init__(self, exp_dir_prefix=None, data_dir_prefix=None) -> None:
        super().__init__(exp_dir_prefix, data_dir_prefix)

    @property
    def approach_name(self) -> str:
        return 'md-cpd'

    def execute(self, data_name, data, including_eval=False, **kwargs):
        return super().execute(data_name=data_name, data=data, including_eval=including_eval, **kwargs)

    def train(self, data, **kwargs):
        mdu.enable_change_point()

        MAX_PROP_ITER = kwargs.get('max_prop_iter', 2)
        MIN_PROP_ITER = kwargs.get('min_prop_iter', 1)
        MIN_LOSS = kwargs.get('min_loss', 1e-6)
        MIN_AT = kwargs.get('min_at', 0.4)
        DECAY_ALPHA = kwargs.get('decay_alpha', 0.)

        optimize = kwargs['optimize']
        num_cp = kwargs['num_cp']
        data_name = kwargs['data_name']
        distance_function = kwargs.get('distance_function', d.AbsD3())

        scale = sum([len(el) for el in data['edge_list']])
        dl = DyLPA(data['num_nodes'],
                   max_iter=MAX_PROP_ITER,
                   min_iter=MIN_PROP_ITER,
                   min_loss=MIN_LOSS,
                   decay_alpha=DECAY_ALPHA,
                   scale_estimated=scale,
                   min_at=MIN_AT)
        distance = DistanceFunction(data_name, distance_function)

        t1 = time.time()
        for i, e_list in enumerate(data['edge_list']):
            if len(e_list) != 0:
                loss_list, pred_result_list, F_list, D_list = dl.add_edges(data['edge_list'][i],
                                                                           ([1] * len(data['edge_list'][i])),
                                                                           check_interval=0,
                                                                           interaction_type=data['type_list'][i],
                                                                           pre_annotation=[], )
                if optimize:
                    mdu.D_intervene_F(D_list, F_list)
                distance.compute_and_store(F_list, is_increment=True)

        t2 = time.time()

        return distance.top_k_predict(num_cp), t2 - t1

class MDCPDSnapshot(ChangePointDetectionAlgorithmComponent):
    def __init__(self, exp_dir_prefix=None, data_dir_prefix=None) -> None:
        super().__init__(exp_dir_prefix, data_dir_prefix)
        self.component = None

    @property
    def approach_name(self) -> str:
        return 'md-cpd'

    def execute(self, data_name, data, including_eval=False, **kwargs):
        return super().execute(data_name=data_name, data=data, including_eval=including_eval, **kwargs)

    def train(self, data, **kwargs):
        mdu.enable_change_point()

        MAX_PROP_ITER = kwargs.get('max_prop_iter', 2)
        MIN_PROP_ITER = kwargs.get('min_prop_iter', 1)
        MIN_LOSS = kwargs.get('min_loss', 1e-6)
        MIN_AT = kwargs.get('min_at', 0.4)
        DECAY_ALPHA = kwargs.get('decay_alpha', 0.)
        time_step2snapshot_id = kwargs['time_step_map']

        optimize = kwargs['optimize']
        num_cp = kwargs['num_cp']
        data_name = kwargs['data_name']
        distance_function = kwargs.get('distance_function', d.AbsD3())

        scale = sum([len(el) for el in data['edge_list']])
        dl = DyLPA(data['num_nodes'],
                   max_iter=MAX_PROP_ITER,
                   min_iter=MIN_PROP_ITER,
                   min_loss=MIN_LOSS,
                   decay_alpha=DECAY_ALPHA,
                   scale_estimated=scale,
                   min_at=MIN_AT)
        self.component = distance = DistanceFunction(data_name, distance_function)

        t1 = time.time()
        for i, e_list in enumerate(data['edge_list']):
            if len(e_list) != 0:
                loss_list, pred_result_list, F_list, D_list = dl.add_edges(data['edge_list'][i],
                                                                           ([1] * len(data['edge_list'][i])),
                                                                           check_interval=0,
                                                                           interaction_type=data['type_list'][i],
                                                                           pre_annotation=[], )
                if optimize:
                    mdu.D_intervene_F(D_list, F_list)
                distance.compute_and_store(F_list, is_increment=True)

        t2 = time.time()

        # 自行实现一个预测方式
        # pred = []
        # copied = distance.stored_change_rate_result.copy()
        # # filter nan and inf
        # for i in range(len(copied)):
        #     if np.isinf(copied[i]) or np.isnan(copied[i]):
        #         copied[i] = 0
        # cp_list = np.argsort(copied, )[::-1]
        # for idx in cp_list:
        #     # predict. combine timestep map and the value
        #     snapshot_id = time_step2snapshot_id[idx]
        #     if snapshot_id not in pred:
        #         pred.append(snapshot_id)
        #     if len(pred) >= num_cp:
        #         break
        pred = self.predict(distance, time_step2snapshot_id, num_cp)

        return pred, t2 - t1

    def predict(self, distance, time_step2snapshot_id, num_cp):
        # 自行实现一个预测方式，从`NormCPDSnapshot`复制过来的
        pred = []
        copied = distance.stored_change_rate_result.copy()
        # filter nan and inf
        for i in range(len(copied)):
            if np.isinf(copied[i]) or np.isnan(copied[i]):
                copied[i] = 0
        cp_list = np.argsort(copied, )[::-1]
        for idx in cp_list:
            # predict. combine timestep map and the value
            snapshot_id = time_step2snapshot_id[idx]
            if snapshot_id not in pred:
                pred.append(snapshot_id)
            if len(pred) >= num_cp:
                break

        return sorted(pred)


class MDCPDLargeScaleCompatibleVersion(MDCPD):
    def train(self, data, **kwargs):
        mdu.enable_change_point()

        MAX_PROP_ITER = kwargs.get('max_prop_iter', 2)
        MIN_PROP_ITER = kwargs.get('min_prop_iter', 1)
        MIN_LOSS = kwargs.get('min_loss', 1e-6)
        MIN_AT = kwargs.get('min_at', 0.4)
        DECAY_ALPHA = kwargs.get('decay_alpha', 0.)

        optimize = kwargs['optimize']
        num_cp = kwargs['num_cp']
        data_name = kwargs['data_name']

        edge_per_snapshot = kwargs.get('edge_per_snapshot', 1000)

        scale = sum([len(el) for el in data['edge_list']])
        dl = DyLPA(data['num_nodes'],
                   max_iter=MAX_PROP_ITER,
                   min_iter=MIN_PROP_ITER,
                   min_loss=MIN_LOSS,
                   decay_alpha=DECAY_ALPHA,
                   scale_estimated=scale,
                   min_at=MIN_AT)
        self.distance = DistanceFunction(data_name, d.AbsD3())

        t1 = time.time()
        use_edge = 0
        for i in range(math.ceil(len(data['edge_list']) / edge_per_snapshot)):
            right_bound = min(use_edge + edge_per_snapshot, len(data['edge_list']))
            loss_list, pred_result_list, F_list, D_list = dl.add_edges(data['edge_list'][use_edge: right_bound],
                                                                       data['weights'][use_edge: right_bound],
                                                                       0.)
            use_edge = right_bound

            if optimize:
                mdu.D_intervene_F(D_list, F_list)
            self.distance.compute_and_store(F_list, is_increment=True)
        t2 = time.time()
        self.dylpa = dl # save it and to draw some graphs
        return self.distance.top_k_predict(num_cp), t2 - t1
