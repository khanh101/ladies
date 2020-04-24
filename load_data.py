import pdb
import numpy as np
from types import SimpleNamespace
from typing import List
import networkx as nx
import scipy.sparse as sparse

def load_random_block(cluster_size: List[int] = [5, 5], prob_matrix: List[List[float]] = [[1, 0],[0, 1]]) -> SimpleNamespace:
    net = nx.generators.community.stochastic_block_model(cluster_size, prob_matrix)
    adj_matrix = nx.adjacency_matrix(net)
    num_nodes = sum(cluster_size)
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    idx_train = idx[:int(0.8 * len(idx))]
    idx_valid = idx[int(0.8 * len(idx)): int(0.9 * len(idx))]
    idx_test = idx[int(0.9 * len(idx)):]

    labels = np.array(list(map(lambda node: net.nodes[node]["block"], net.nodes)))
    eye = np.eye(num_nodes)
    np.random.shuffle(eye)
    features = sparse.csr_matrix(eye)
    #features = sparse.csr_matrix(sparse.eye(num_nodes))
    
    data = SimpleNamespace(
        adj_matrix= adj_matrix,
        train_nodes= idx_train,
        valid_nodes= idx_valid,
        test_nodes=idx_test,
        labels= labels,
        features= features,
    ) 

    return data
if __name__ == "__main__":
    data = load_random_block()
    num_nodes = data.adj_matrix.shape[0]
    import networkx as nx
    import matplotlib.pyplot as plt
    net = nx.from_numpy_array(data.adj_matrix.toarray())
    nx.draw(net, node_size= 10)
    plt.show()


    import pdb
    pdb.set_trace()
