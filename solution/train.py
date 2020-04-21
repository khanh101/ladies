#!/usr/bin/env python
import pdb
import argparse
from typing import List, Dict
from types import SimpleNamespace
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import GCN, Classifier
from load_data import load



if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Training GCN')
  parser.add_argument('--nhid', type=int, default=256,
                      help='Hidden layer dimension')
  parser.add_argument('--epoch_num', type=int, default= 100,
                      help='Number of Epoch')
  parser.add_argument('--batch_num', type=int, default= 10,
                      help='Maximum Batch Number')
  parser.add_argument('--batch_size', type=int, default=512,
                      help='size of output node in a batch')
  parser.add_argument('--n_layers', type=int, default=5,
                      help='Number of GCN layers')
  parser.add_argument('--n_iters', type=int, default=2,
                      help='Number of iteration to run on a batch')
  parser.add_argument('--samp_num', type=int, default=64,
                      help='Number of sampled nodes per layer')
  parser.add_argument('--sample_method', type=str, default='default',
                      help='Sampled Algorithms: default/ladies')
  parser.add_argument('--cuda', type=int, default=0,
                      help='Avaiable GPU ID')
  parser.add_argument('--pool_num', type=int, default= 1,
                    help='Number of Pool Processes')

  args = parser.parse_args()
  print(args)
  # load data
  adj_matrix, train_nodes, valid_nodes, test_nodes, edges, labels, feat_data, num_classes = load()

  model: SimpleNamespace = SimpleNamespace()
  model.pool = mp.Pool(processes= args.pool_num)

  print(add(3))
  model.add = add
  value = model.pool.apply_async(model.add, args=(3,))

  print(value.get()) 
  
  pdb.set_trace()
def add(x):
    return x+1