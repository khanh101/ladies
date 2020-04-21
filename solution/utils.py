from typing import Tuple

import torch
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import sys
import pickle as pkl

def adj_to_lap_matrix(adj_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    lap_matrix = row_normalize(adj_matrix + sparse.csr_matrix(np.eye(adj_matrix.shape[0])))
    return lap_matrix

def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sparse.csr_matrix) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
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
