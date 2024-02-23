import os
import time

import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds

from src.baselines.cpd_baseilne import ChangePointDetectionAlgorithmComponent
from src.utils.lad_utils import lad_util
from src.utils.lad_utils.Anomaly_Detection import detection_with_bothwindows


class LAD(ChangePointDetectionAlgorithmComponent):
    def __init__(self, exp_dir_prefix=None, data_dir_prefix=None) -> None:
        super().__init__(exp_dir_prefix, data_dir_prefix)

    @property
    def approach_name(self) -> str:
        return 'lad'

    def get_exp_save_dir(self, data_name):
        return os.path.join(self.exp_dir_prefix, data_name, self.approach_name)

    def SVD_perSlice(self, G_times, directed=True, num_eigen=6, top=True, max_size=500):
        Temporal_eigenvalues = []
        activity_vecs = []  # eigenvector of the largest eigenvalue
        counter = 0

        for G in G_times:
            if len(G) < max_size:
                for i in range(len(G), max_size):
                    G.add_node(-1 * i)  # add empty node with no connectivity (zero padding)
            if directed:
                L = nx.directed_laplacian_matrix(G)

            else:
                L = nx.laplacian_matrix(G)
                L = L.asfptype()

            if top:
                which = "LM"
            else:
                which = "SM"

            u, s, vh = svds(L, k=num_eigen, which=which)
            vals = s
            vecs = u
            max_index = list(vals).index(max(list(vals)))
            activity_vecs.append(np.asarray(vecs[max_index]))
            Temporal_eigenvalues.append(np.asarray(vals))

            print("processing " + str(counter), end="\r")
            counter = counter + 1

        return Temporal_eigenvalues, activity_vecs

    def execute(self, data_name, data, including_eval=False, **kwargs):
        return super().execute(data_name=data_name, data=data, including_eval=including_eval, **kwargs)

    def train(self, data, **kwargs):
        # ------------ SVD --------------
        G_times = data
        directed = kwargs.get('directed', True)
        num_eigen = kwargs.get('num_eigen', 6)
        top = kwargs.get('top', True)

        num_nodes = kwargs['num_nodes']

        t1 = time.time()
        (Temporal_eigenvalues, activity_vecs) = self.SVD_perSlice(G_times, directed=directed,
                                                                  num_eigen=num_eigen, top=top,
                                                                  max_size=num_nodes)

        save_dir = self.get_exp_save_dir(kwargs['data_name'])
        save_path = os.path.join(save_dir, 'singular.pkl')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        lad_util.save_object(Temporal_eigenvalues, save_path)

        # -------------- detection ---------------
        timestamps = len(G_times)
        percent_ranked = 0.20
        eigen_file = save_path
        difference = True

        window1 = 1
        window2 = 2
        initial_window = 2
        (z_shorts, z_longs, z_scores, events) = detection_with_bothwindows(eigen_file=eigen_file, timestamps=timestamps,
                                                                           percent_ranked=percent_ranked,
                                                                           window1=window1,
                                                                           window2=window2,
                                                                           initial_window=initial_window,
                                                                           difference=difference)

        t2 = time.time()

        # return z_shorts, z_longs, z_scores, events, t2 - t1
        return events, t2 - t1
