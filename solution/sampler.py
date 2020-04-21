
import numpy as np
import scipy
from typing import Tuple
from utils import sparse_mx_to_torch_sparse_tensor, row_normalize

"""
def default_sampler(seed: int, batch_nodes: np.ndarray, samp_num_list: nd.ndarray, num_nodes: int, lap_matrix: sp.sparse.spmatrix, depth: int) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    seed: seed
    batch_nodes: 1d array of nodes at output layer
    samp_num_list: 
"""

def default_sampler(seed: int, batch_nodes: np.ndarray, samp_num_list: nd.ndarray, num_nodes: int, lap_matrix: sp.sparse.spmatrix, depth: int) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx for i in range(depth)], np.arange(num_nodes), batch_nodes

def ladies_sampler(seed: int, batch_nodes: np.ndarray, samp_num_list: nd.ndarray, num_nodes: int, lap_matrix: sp.sparse.spmatrix, depth: int) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        """ NEW
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :] #square laplacian matrix
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.sum(U, axis=0)
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        """

        # ORIGINAL: THESE LINES OF CODE MIGHT BE WRONG
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U, axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])



        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

if __name__ == "__main__":
  pass