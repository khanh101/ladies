import pdb
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import torch
from types import SimpleNamespace
from typing import Tuple, List
from utils import row_normalize, sparse_fill

def sampler(batch_nodes: np.ndarray, samp_num_list: np.ndarray, num_nodes: int, p_matrix: sparse.csr_matrix, lap2_matrix: sparse.csr_matrix, num_layers: int) -> SimpleNamespace:
  ##### INPUT ######################################################
  # batch_nodes : sampled nodes at output
  # samp_num_list: array of number of sampled nodes at all layers
  # num_nodes : number of graph nodes
  # p_matrix : sampled matrix 
  # lap2_matrix : squared lap_matrix (precomputed)
  # num_layers : len(samp_num_list)
  ##### OUTPUT ######################################################
  # adjs : P matrix
  # input_nodes: sampled nodes at input
  # output_nodes : batch_nodes
  ###################################################################
  raise Exception("sampler must not be called")

def full_sampler(batch_nodes: np.ndarray, samp_num_list: np.ndarray, num_nodes: int, p_matrix: sparse.csr_matrix, lap2_matrix: sparse.csr_matrix, num_layers: int) -> SimpleNamespace:
  # simply sample the full p_matrix for every layers
  sample = SimpleNamespace(
    adjs= [p_matrix for _ in range(num_layers)],
    input_nodes= np.arange(num_nodes),
    output_nodes= batch_nodes,
  )
  return sample

def ladies_sampler(batch_nodes: np.ndarray, samp_num_list: np.ndarray, num_nodes: int, p_matrix: sparse.csr_matrix, lap2_matrix: sparse.csr_matrix, num_layers: int) -> SimpleNamespace:
  '''
    LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
             is computed adaptively according to the nodes sampled in the upper layer.
  '''
  previous_nodes = batch_nodes
  adjs  = []
  '''
    Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
  '''
  for d in range(num_layers):
    # row-select the lap2_matrix (U2) by previously sampled nodes
    U2 = lap2_matrix[previous_nodes , :]
    # calculate sampling probablities
    pi = np.sum(U2, axis=0)
    p = pi / np.sum(pi)
    # get number of sampled nodes
    s_num = np.min([np.sum(p > 0), samp_num_list[d]])
    p = p.view(np.ndarray).flatten()
    # sample the next layer's nodes based on the adaptively probability (p).
    after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
    # Add output nodes for self-loop
    after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
    # row-select and col-select the p_matrix (U), and then devided by the sampled probability for unbiased-sampling.
    adj = p_matrix[previous_nodes, :][:, after_nodes]
    adj = adj.multiply(1/ p[after_nodes])
    # conduct row-normalization to avoid value explosion.    
    adj = row_normalize(adj)
    # fill the sub-matrix into the original
    adj = sparse_fill(p_matrix.shape, adj, previous_nodes, after_nodes)
    adjs.append(adj)
    # turn the sampled nodes as previous_nodes, recursively conduct sampling.
    previous_nodes = after_nodes
  # Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
  adjs.reverse()

  sample = SimpleNamespace(
    adjs= adjs,
    input_nodes= previous_nodes,
    output_nodes= batch_nodes,
  )
  return sample
if __name__ == "__main__":
  pass