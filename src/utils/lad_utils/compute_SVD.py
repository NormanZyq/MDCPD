import os.path

import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds

from src.utils.lad_utils import rdyn_loader, lad_util

'''
compute the eigenvalues for square laplacian matrix per time slice 
input: list of networkx Graphs
output: list of 1d numpy array of diagonal entries computed from SVD
'''

def SVD_perSlice(G_times, directed=True, num_eigen=6, top=True, max_size=500):
    Temporal_eigenvalues = []
    activity_vecs = []  # eigenvector of the largest eigenvalue
    counter = 0

    for G in G_times:
        if (len(G) < max_size):
            for i in range(len(G), max_size):
                G.add_node(-1 * i)  # add empty node with no connectivity (zero padding)
        if (directed):
            L = nx.directed_laplacian_matrix(G)

        else:
            L = nx.laplacian_matrix(G)
            L = L.asfptype()

        if (top):
            which = "LM"
        else:
            which = "SM"

        u, s, vh = svds(L, k=num_eigen, which=which)
        # u, s, vh = randomized_svd(L, num_eigen)
        vals = s
        vecs = u
        # vals, vecs= LA.eig(L)
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues.append(np.asarray(vals))

        print("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)


def compute_rdyn_SVD(data_name, directed=True, num_eigen=100, top=True):
    fname = 'data/{}/data.pkl'.format(data_name)
    G_times = rdyn_loader.load_edgelist(fname)
    num_nodes = 500
    (Temporal_eigenvalues, activity_vecs) = SVD_perSlice(G_times, directed=directed, num_eigen=num_eigen, top=top,
                                                         max_size=num_nodes)
    save_dir = 'experiments/cpd/lad/{}'.format(data_name)
    save_path = save_dir + '/singular.pkl'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    lad_util.save_object(Temporal_eigenvalues, save_path)
