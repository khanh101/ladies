from typing import Tuple

import torch
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import sys
import pickle as pkl

def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sparse.spmatrix) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx: np.ndarray = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def load_data(dataset_str):
    if dataset_str == 'ppi':
        prefix = './ppi/ppi'
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)
        if isinstance(G.nodes()[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n

        if os.path.exists(prefix + "-feats.npy"):
            feats = np.load(prefix + "-feats.npy")
        else:
            print("No features present.. Only identity features will be used.")
            feats = None
        id_map = json.load(open(prefix + "-id_map.json"))
        id_map = {conversion(k):int(v) for k,v in id_map.items()}
        walks = []
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)

        class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

        ## Remove all nodes that do not have val/test annotations
        ## (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        for node in G.nodes():
            if not 'val' in G.node[node] or not 'test' in G.node[node]:
                G.remove_node(node)
                broken_count += 1
        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

        ## Make sure the graph has edge train_removed annotations
        ## (some datasets might already have this..)
        print("Loaded data.. now preprocessing..")
        for edge in G.edges():
            if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False

        train_ids = np.array([id_map[str(n)] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        features = scaler.transform(feats)
        
        degrees = np.zeros(len(G), dtype=np.int64)
        edges = []
        labels = []
        idx_train = []
        idx_val   = []
        idx_test  = []
        for s in G:
            if G.nodes[s]['test']:
                idx_test += [s]
            elif G.nodes[s]['val']:
                idx_val += [s]
            else:
                idx_train += [s]
            for t in G[s]:
                edges += [[s, t]]
            degrees[s] = len(G[s])
            labels += [class_map[str(s)]]
        
        return np.array(edges), np.array(degrees), np.array(labels), np.array(features),\
                np.array(idx_train), np.array(idx_val), np.array(idx_test)
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
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

    return np.array(edges), degrees, labels, features,  idx_train, idx_val, idx_test