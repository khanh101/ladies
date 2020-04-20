#!/usr/bin/env python
import pdb
import argparse

import torch
import torch.nn as nn

from module import GCN
from utils import load_data


class Trainer(object):
  def __init__(self, module: nn.Module):
    pass
  def train(x):
    pass



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

  args = parser.parse_args()

  gcn = GCN(in_features= 5, hidden_features= 5, out_features= 5, num_layers= 5, dropout= 0.5)
  # load data
  edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data('cora')
  pdb.set_trace()