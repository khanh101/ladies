import pdb
import sys
import numpy as np
from types import SimpleNamespace
from typing import List
import networkx as nx
import scipy.sparse as sparse
import pickle as pkl

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_adj(edges, num_nodes):
    adj = sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

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

def load_citeseer():
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./dataset/ind.{}.{}".format('citeseer', names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format('citeseer'))
    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

    features = sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = np.array(test_idx_range.tolist())
    idx_train = np.array(range(len(y)))
    idx_val = np.array(range(len(y), len(y)+500))


    edges = []
    for s in graph:
        for t in graph[s]:
            edges += [[s, t]]

    adj_matrix = get_adj(np.array(edges), labels.shape[0])
    idx_train, idx_test = idx_test, idx_train

    labels = onehot_to_labels(labels)

    data = SimpleNamespace(
        adj_matrix= adj_matrix,
        train_nodes= idx_train,
        valid_nodes= idx_val,
        test_nodes=idx_test,
        labels= labels,
        features= features,
    ) 

    return data

def load_random_block(cluster_size: List[int] = [5, 5], prob_matrix: List[List[float]] = [[1, 0],[0, 1]]):
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
    net = nx.from_numpy_array(data.adj_matrix.toarray())
    nx.draw(net, node_size= 10)
    plt.show()


    import pdb
    pdb.set_trace()
