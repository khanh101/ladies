import pdb
from typing import Tuple

import torch
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import sys
import pickle as pkl

def adj_to_lap_matrix(adj_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    lap_matrix = adj_matrix + sparse.eye(adj_matrix.shape[0])
    return lap_matrix

def sparse_fill(shape: np.ndarray, mx: sparse.csr_matrix, row: np.ndarray = None, col: np.ndarray= None) -> sparse.csr_matrix:
    if row is None:
        row = np.arange(shape[0])
    if col is None:
        col = np.arange(shape[1])
    """
    dense = np.zeros(shape)
    mxdense = mx.toarray()
    for r, rr in enumerate(row):
        for c, cc in enumerate(col):
            dense[rr][cc] = mxdense[r][c]
    return sparse.csr_matrix(dense)
    """
    lil = sparse.lil_matrix(mx.shape)
    for r, rr in enumerate(row):
        for c, cc in enumerate(col):
            lil[rr, cc] = mx[r, c]
    return sparse.csr_matrix(lil)

def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum == 0] = 1 # rowsum -> no need to divide
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sparse.csr_matrix) -> torch.FloatTensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
