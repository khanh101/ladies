import numpy as np
from types import SimpleNamespace
from typing import List
import scipy.sparse as sparse
import networkx as nx


def load_random_block(cluster_size: List[int] = [5, 5],
                      prob_matrix: List[List[float]] = [[1, 0], [0, 1]]) -> SimpleNamespace:
    net = nx.generators.community.stochastic_block_model(cluster_size, prob_matrix)
    adj_matrix = nx.adjacency_matrix(net).astype(np.float32)
    num_nodes = sum(cluster_size)
    idx = np.random.permutation(num_nodes)
    idx_train = idx[:int(0.8 * len(idx))]
    idx_valid = idx[int(0.8 * len(idx)):]

    labels = np.array(list(map(lambda node: net.nodes[node]["block"], net.nodes)))

    features = sparse.lil_matrix((num_nodes, num_nodes))
    a_range = np.random.permutation(num_nodes)
    for i in range(num_nodes):
        features[i, a_range[i]] = 1
    features = sparse.csr_matrix(features)

    data = SimpleNamespace(
        adj_matrix=adj_matrix,
        train_nodes=idx_train,
        valid_nodes=idx_valid,
        labels=labels,
        features=features,
    )

    return data

