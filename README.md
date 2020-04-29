# OVERVIEW

A correct version of 

`Code for NeurIPS'19 "Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks"`

latest update can be found at [here](https://github.com/khanhhhh/ladies)

Original code [here](https://github.com/acbull/LADIES)

# HOW TO RUN

```
usage: train.py [-h] [--hidden_features HIDDEN_FEATURES]
                [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--sampled_size SAMPLED_SIZE] [--num_layers NUM_LAYERS]
                [--sampling_method SAMPLING_METHOD] [--cuda CUDA]
                [--dropout DROPOUT] [--learning_rate LEARNING_RATE]
                [--num_nodes NUM_NODES] [--p_matrix P_MATRIX]

Training GCN

optional arguments:
  -h, --help            show this help message and exit
  --hidden_features HIDDEN_FEATURES
                        hidden layer embedding dimension
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --batch_size BATCH_SIZE
                        batch_size: number of sampled nodes at output layer
  --sampled_size SAMPLED_SIZE
                        sampled_size: number of sampled nodes at hidden layers
  --num_layers NUM_LAYERS
                        Number of GCN layers
  --sampling_method SAMPLING_METHOD
                        sampling algorithms: full/ladies
  --cuda CUDA           GPU ID, (-1) for CPU
  --dropout DROPOUT     dropout probability
  --learning_rate LEARNING_RATE
                        learning rate
  --num_nodes NUM_NODES
                        number of network nodes in random block model
  --p_matrix P_MATRIX   p_matrix: normalized_laplacian/normalized_adjacency/be
                        the_hessian
```

- Example
```
python train.py --num_nodes 1024 --num_layers 5 --sampling_method ladies --dropout 0.2
```

# FILE STRUCTURE

- `load_data.py` : code for random-block graph generation

-  `module.py` : code for GCN module

- `sampler.py`: code for node sampling

- `train.py` : code for training

- `utils.py` : code for other functions (converting adj to lap matrix, fill a sparse matrix, row normalization, scipy sparse matrix to pytorch sparse matrix)

- `README.md`

# CREDITS 

This code uses some of the functions taken from 

- `https://github.com/tkipf/pygcn` : GCN layer implementation

- `https://github.com/acbull/LADIES`: sparse matrix row-normalization, converting scipy sparse matrix to pytorch sparse matrix
