#!/usr/bin/env python
import pdb
import argparse
from typing import List, Dict
from types import SimpleNamespace
import multiprocessing as mp

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import GCN, Classifier
from load_data import load
from sampler import full_sampler, ladies_sampler
from utils import adj_to_lap_matrix, row_normalize

def random_sampling_train(args: SimpleNamespace, model: SimpleNamespace, data: SimpleNamespace) -> SimpleNamespace:
  batch_nodes = np.random.choice(data.train_nodes, size= args.batch_size)
  sample = model.sampler(
    batch_nodes= batch_nodes,
    samp_num_list= [args.batch_size for _ in range(args.num_layers)],
    num_nodes= data.num_nodes,
    lap2_matrix= data.lap2_matrix,
    num_layers= args.num_layers,
  )
  return sample


if __name__ == "__main__":
  # load args
  parser = argparse.ArgumentParser(description='Training GCN')
  parser.add_argument('--hidden_features', type=int, default=256,
                      help='Hidden layer dimension')
  parser.add_argument('--num_epochs', type=int, default= 100,
                      help='Number of Epoch')
  parser.add_argument('--batch_size', type=int, default=64,
                      help='size of output node in a batch')
  parser.add_argument('--num_layers', type=int, default=5,
                      help='Number of GCN layers')
  parser.add_argument('--num_iterations', type=int, default=2,
                      help='Number of iteration to run on a batch')
  parser.add_argument('--sampling_method', type=str, default='full',
                      help='Sampled Algorithms: full/ladies')
  parser.add_argument('--cuda', type=int, default=0,
                      help='Avaiable GPU ID')
  parser.add_argument('--num_processes', type=int, default= 1,
                    help='Number of Pool Processes')
  parser.add_argument('--dropout', type=float, default= 0.5,
                    help='Dropout probability')
  parser.add_argument('--seed', type=int, 
                    help='Random Seed')

  args = parser.parse_args()
  
  # load data
  data = load()
  data.num_nodes = data.features.shape[0]
  data.in_features = data.features.shape[1]
  data.out_features = len(np.unique(data.labels))
  data.lap_matrix = adj_to_lap_matrix(data.adj_matrix)
  lap_norm_matrix = row_normalize(data.lap_matrix)
  data.lap2_matrix = np.multiply(lap_norm_matrix, lap_norm_matrix)
  # create model
  model: SimpleNamespace = SimpleNamespace()
  model.pool = mp.Pool(processes= args.num_processes)
  model.cuda = args.cuda
  model.sampler = full_sampler
  if args.sampling_method == "ladies":
    model.sampler = ladies_sampler
  encoder = GCN(
    in_features= data.in_features,
    hidden_features= args.hidden_features,
    out_features= args.hidden_features,
    num_layers= args.num_layers,
    dropout= args.dropout
  )
  model.module = Classifier(
    encoder= encoder,
    in_features= args.hidden_features,
    out_features= data.out_features,
    dropout= args.dropout,
  )

  sample = random_sampling_train(args, model, data)

  pdb.set_trace()