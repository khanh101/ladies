import numpy as np
import sys
import scipy.sparse as sp
import pickle as pkl

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_adj(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

def load():
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format('citeseer', names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format('citeseer'))
    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = np.array(test_idx_range.tolist())
    idx_train = np.array(range(len(y)))
    idx_val = np.array(range(len(y), len(y)+500))


    degrees = np.zeros(len(labels), dtype=np.int64)
    edges = []
    for s in graph:
        for t in graph[s]:
            edges += [[s, t]]
        degrees[s] = len(graph[s])

    adj_matrix = get_adj(np.array(edges), labels.shape[0])
    tmp = idx_test
    idx_test = idx_train
    idx_train = tmp

    return   adj_matrix, idx_train, idx_val, idx_test, np.array(edges), degrees, labels, features
if __name__ == "__main__":
    adj_matrix, train_nodes, valid_nodes, test_nodes, edges, labels, feat_data, num_classes = load()
    print(train_nodes.shape)
    print(valid_nodes.shape)
    print(test_nodes.shape)
    print(adj_matrix.shape)
    print(edges.shape)
    print(labels.shape)
    print(feat_data.shape)
    print(num_classes.shape)
    import pdb
    pdb.set_trace()
