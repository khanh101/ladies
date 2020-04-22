import pdb
import numpy as np
from types import SimpleNamespace
from typing import List
import networkx as nx
import scipy.sparse as sparse

def onehot_to_labels(onehot: np.ndarray) -> np.ndarray:
  out = []
  no_label = 0
  for row in onehot:
    where = np.where(row==1)
    if len(where[0]) == 1:
      out.append(where[0][0])
    else:
      no_label += 1
      out.append(0)
  print(f" nolabel {no_label} ", end="")
  return np.array(out)
  return np.array([np.where(row==1)[0][0] for row in onehot])

def load(cluster_size: List[int] = [5, 5], prob_matrix: List[List[float]] = [[1, 0],[0, 1]]):
    net = nx.generators.community.stochastic_block_model(cluster_size, prob_matrix)
    adj_matrix = nx.adjacency_matrix(net)
    num_nodes = sum(cluster_size)
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    idx_train = idx[:int(0.8 * len(idx))]
    idx_valid = idx[int(0.8 * len(idx)): int(0.9 * len(idx))]
    idx_test = idx[int(0.9 * len(idx)):]

    labels = np.array(list(map(lambda node: net.nodes[node]["block"], net.nodes)))
    features = sparse.csr_matrix(sparse.eye(num_nodes))
    
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
    data = load()
    num_nodes = data.adj_matrix.shape[0]
    import networkx as nx
    import matplotlib.pyplot as plt
    net = nx.from_numpy_matrix(data.adj_matrix.todense())
    nx.draw(net, node_size= 10)
    plt.show()


    import pdb
    pdb.set_trace()
