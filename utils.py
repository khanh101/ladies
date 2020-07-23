from typing import Tuple

import torch
import numpy as np
import scipy.sparse as sparse


def adj_to_deg_matrix(adj_matrix: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Adjacency matrix to Degree matrix
    """
    deg_in = sparse.lil_matrix(adj_matrix.shape)
    deg_out = sparse.lil_matrix(adj_matrix.shape)

    row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
    col_sum = np.array(adj_matrix.sum(axis=0)).flatten()
    for i in range(adj_matrix.shape[0]):
        deg_in[i, i] = row_sum[i]
        deg_out[i, i] = col_sum[i]
    return sparse.csr_matrix(deg_in), sparse.csr_matrix(deg_out)


def adj_to_lap_matrix(adj_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Adjacency matrix to Laplacian matrix
    """
    lap_matrix = adj_matrix + sparse.eye(adj_matrix.shape[0])
    return lap_matrix


def sparse_fill(shape: np.ndarray, mx: sparse.csr_matrix, row: np.ndarray = None,
                col: np.ndarray = None) -> sparse.csr_matrix:
    """
    Fill a (m, n) matrix into m x n entries in a (M, N) matrix
    shape : (M, N)
    mx: (m, n) matrix
    row, col: filled-in rows and columns
    """
    if row is None:
        row = np.arange(shape[0])
    if col is None:
        col = np.arange(shape[1])
    lil = sparse.lil_matrix(shape)
    mx = mx.toarray()
    for r, rr in enumerate(row):
        for c, cc in enumerate(col):
            lil[rr, cc] = mx[r, c]
    return sparse.csr_matrix(lil)


def row_normalize(mx: sparse.csr_matrix):
    """
    Row-normalize a matrix
    """
    row_sum = np.array(mx.sum(axis=1))
    row_sum[row_sum == 0] = 1  # row_sum == 0 -> no need to divide
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(mx: sparse.csr_matrix) -> torch.FloatTensor:
    """
    Convert a scipy sparse matrix to a pytorch sparse tensor.
    """
    mx = mx.tocoo().astype(np.float32)
    if len(mx.row) == 0 and len(mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
