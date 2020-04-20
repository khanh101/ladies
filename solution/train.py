#!/usr/bin/env python

import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

  '''
      Dataset arguments
  '''
  parser.add_argument('--dataset', type=str, default='reddit',
                      help='Dataset name: Cora/CiteSeer/PubMed/Reddit')
  parser.add_argument('--nhid', type=int, default=256,
                      help='Hidden state dimension')
  parser.add_argument('--epoch_num', type=int, default= 100,
                      help='Number of Epoch')
  parser.add_argument('--pool_num', type=int, default= 10,
                      help='Number of Pool')
  parser.add_argument('--batch_num', type=int, default= 10,
                      help='Maximum Batch Number')
  parser.add_argument('--batch_size', type=int, default=512,
                      help='size of output node in a batch')
  parser.add_argument('--n_layers', type=int, default=5,
                      help='Number of GCN layers')
  parser.add_argument('--n_iters', type=int, default=2,
                      help='Number of iteration to run on a batch')
  parser.add_argument('--n_stops', type=int, default=200,
                      help='Stop after number of batches that f1 dont increase')
  parser.add_argument('--samp_num', type=int, default=64,
                      help='Number of sampled nodes per layer')
  parser.add_argument('--sample_method', type=str, default='ladies',
                      help='Sampled Algorithms: ladies/fastgcn/full')
  parser.add_argument('--cuda', type=int, default=0,
                      help='Avaiable GPU ID')



  args = parser.parse_args()