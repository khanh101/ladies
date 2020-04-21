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
import torch.optim as optim
from sklearn.metrics import f1_score

from module import GCN, GCNLinear
from load_data import load
from sampler import full_sampler, ladies_sampler
from utils import adj_to_lap_matrix, row_normalize, sparse_mx_to_torch_sparse_tensor, sparse_fill

#np.seterr(all="raise")

def random_sampling_train(args: SimpleNamespace, model: SimpleNamespace, data: SimpleNamespace) -> SimpleNamespace:
  batch_nodes = np.random.choice(data.train_nodes, size= args.batch_size)
  sample = ladies_sampler(
    batch_nodes= batch_nodes,
    samp_num_list= [args.batch_size for _ in range(args.num_layers)],
    num_nodes= data.num_nodes,
    lap_matrix= data.lap_matrix,
    lap2_matrix= data.lap2_matrix,
    num_layers= args.num_layers,
  )
  return sample

def sampling_valid(args: SimpleNamespace, model: SimpleNamespace, data: SimpleNamespace) -> SimpleNamespace:
  sample = full_sampler(
    batch_nodes= data.valid_nodes,
    samp_num_list= None,
    num_nodes= data.num_nodes,
    lap_matrix= data.lap_matrix,
    lap2_matrix= None,
    num_layers= args.num_layers,
  )
  return sample

def onehot_to_labels(onehot: np.ndarray) -> np.ndarray:
  return np.array([np.where(row==1)[0][0] for row in onehot])


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
  parser.add_argument('--num_iterations', type=int, default=1,
                      help='Number of iteration to run on a batch')
  parser.add_argument('--sampling_method', type=str, default='full',
                      help='Sampled Algorithms: full/ladies')
  parser.add_argument('--cuda', type=int, default=-1,
                      help='Avaiable GPU ID')
  #parser.add_argument('--num_processes', type=int, default= 1,
  #                  help='Number of Pool Processes')
  parser.add_argument('--dropout', type=float, default= 0.5,
                    help='Dropout probability')
  #parser.add_argument('--seed', type=int, 
  #                  help='Random Seed')

  args = parser.parse_args()
  print(args)
  # set up device
  if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
  else:
    device = torch.device("cpu")
  # load data
  data = load()
  data.num_nodes = data.features.shape[0]
  data.in_features = data.features.shape[1]
  data.out_features = data.labels.shape[1]
  data.lap_matrix = row_normalize(adj_to_lap_matrix(data.adj_matrix))
  data.lap2_matrix = np.multiply(data.lap_matrix, data.lap_matrix)
  # create pool
  pool = mp.Pool(processes= 1)
  # create model
  model: SimpleNamespace = SimpleNamespace()
  model.sampler = full_sampler
  if args.sampling_method == "ladies":
    model.sampler = ladies_sampler
  model.module = GCNLinear(
    encoder= GCN(
      in_features= data.in_features,
      hidden_features= args.hidden_features,
      out_features= args.hidden_features,
      num_layers= args.num_layers,
      dropout= args.dropout
    ),
    in_features= args.hidden_features,
    out_features= data.out_features,
    dropout= args.dropout,
  )

  model.module.to(device)
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=1e-2)
  criterion = nn.CrossEntropyLoss()
  losses = []
  next_sample_async = None
  for epoch in range(args.num_epochs):
    # train
    model.module.train() # train mode
    print(f"Epoch {epoch}: ", flush= True)
    for iter in range(args.num_iterations):
      print(f"\tIteration {iter}: ", end= "", flush= True)
      if next_sample_async is None:
        sample = random_sampling_train(args, model, data)
      else:
        sample = next_sample_async.get()
      next_sample_async = pool.apply_async(random_sampling_train, args= (args, model, data))
      optimizer.zero_grad()
      output = model.module(
        x= sparse_mx_to_torch_sparse_tensor(sparse_fill(shape= data.features.shape, mx= data.features[sample.input_nodes], row= sample.input_nodes)),
        adjs= list(map(lambda adj: sparse_mx_to_torch_sparse_tensor(adj).to(device), sample.adjs)),
      )
      loss = criterion(
        output[sample.output_nodes],
        torch.from_numpy(onehot_to_labels(data.labels[sample.output_nodes])).long(),
      )
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.module.parameters(), 0.2)
      optimizer.step()
      loss = loss.detach().cpu()
      print(f"Loss {loss}", flush= True)
    # eval
    model.module.eval() # eval mode
    sample = sampling_valid(args, model, data)
    output = model.module(
      x= sparse_mx_to_torch_sparse_tensor(data.features[sample.input_nodes]),
      adjs= list(map(lambda adj: sparse_mx_to_torch_sparse_tensor(adj).to(device), sample.adjs)),
    )
    loss = criterion(
      output[sample.output_nodes],
      torch.from_numpy(onehot_to_labels(data.labels[sample.output_nodes])).long(),
    )
    output = output.detach().cpu()
    loss = loss.detach().cpu()
    f1 = f1_score(
      output[sample.output_nodes].argmax(dim=1),
      onehot_to_labels(data.labels[sample.output_nodes]),
      average= "micro",
    )
    losses.append(loss)
    print(f"Epoch {epoch}: Loss {loss} F1 {f1}", flush= True)

  import matplotlib.pyplot as plt
  plt.plot(np.arange(len(losses)), losses)
  plt.show()
  pdb.set_trace()