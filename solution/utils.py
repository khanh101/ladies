from typing import Tuple

import torch
import numpy as np
import scipy


def sparse_mx_to_torch_sparse_tensor(sparse_mx: scipy.sparse.spmatrix) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
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