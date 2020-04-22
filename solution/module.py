

from typing import List


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLinear(nn.Module):
  def __init__(self, encoder: nn.Module, in_features: int, out_features: int, dropout: float):
    super(GCNLinear, self).__init__()
    self.encoder = encoder
    self.dropout = nn.Dropout(p= dropout)
    self.linear = nn.Linear(in_features= in_features, out_features= out_features, bias= True)
  def forward(self, x: torch.Tensor, adjs: List[torch.Tensor]) -> torch.Tensor:
    x = self.encoder(x= x, adjs= adjs)
    x = self.dropout(x)
    x = self.linear(x)
    return x


class GCN(nn.Module):
  def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int, dropout: float):
    super(GCN, self).__init__()
    self.in_features: int = in_features
    self.out_features: int = out_features
    self.gcs: nn.ModuleList = nn.ModuleList()
    self.dropout: float = dropout
    if num_layers >= 2:
      self.gcs.append(GraphConvolution(in_features= in_features, out_features= hidden_features))
      for i in range(num_layers-2):
        self.gcs.append(GraphConvolution(in_features= hidden_features, out_features= hidden_features))
      self.gcs.append(GraphConvolution(in_features= hidden_features, out_features= out_features))
    else:
      self.gcs.append(GraphConvolution(in_features= in_features, out_features= out_features))
  
  def forward(self, x: torch.Tensor, adjs: List[torch.Tensor]) -> torch.Tensor:
    for l in range(len(self.gcs)):
      x = self.gcs[l](x, adjs[l])
      if l != len(self.gcs)-1:
        x = F.dropout(F.relu(x), p= self.dropout, training= self.training)
      else: # no dropout and relu for the last layer
        x = x
    return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features: int, out_features: int, bias: bool =True):
        super(GraphConvolution, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: nn.Parameter = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias: nn.Parameter = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

if __name__ == "__main__":
  gcn = GCN(in_features= 10, hidden_features= 5, out_features = 5, num_layers = 5, dropout= 0.5)
