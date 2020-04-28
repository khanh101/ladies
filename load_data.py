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
  idx = np.random.permutation(num_nodes)
  idx_train = idx[:int(0.8 * len(idx))]
  idx_valid = idx[int(0.8 * len(idx)):]

  labels = np.array(list(map(lambda node: net.nodes[node]["block"], net.nodes)))

  features = sparse.lil_matrix((num_nodes, num_nodes))
  arange = np.random.permutation(num_nodes)
  for i in range(num_nodes):
    features[i, arange[i]] = 1
  features = sparse.csr_matrix(features)
  
  data = SimpleNamespace(
    adj_matrix= adj_matrix,
    train_nodes= idx_train,
    valid_nodes= idx_valid,
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
