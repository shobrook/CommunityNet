# CommunityNet

`CommunityNet` is a hierarchical Graph Neural Network (GNN) designed for graph datasets with community structure (e.g. social networks, molecules, etc.).<!--It takes an input graph and creates vector embeddings of each community, then combines these embeddings into a inter-community graph, and feeds this graph through a GNN to generate a graph-level prediction.-->

<img src="demo.png" />

## Installation

You can download `CommunityNet` from PyPi:

```bash
$ pip install communitynet
```

## Usage

Each graph you submit to `CommunityNet` must be an instance of `torch_geometric.data.Data` with an additional `communities` attribute. `data.communities` should hold a list of communities, where each community is a set of node indices. Every graph in your dataset must have the same number of communities.

Before instantiating `CommunityNet`, you must define a "base" GNN and an "output" GNN. The _base GNN_ is used to create vector embeddings of each community in an input graph. These embeddings are then used as node features in an inter-community graph, which is submitted to the _output GNN_ to make a prediction. Both GNNs can be constructed using the `GraphNet` and `MLP` PyTorch modules supplied by the library. For example, to construct the `CommunityNet` shown in the diagram above, you can do the following:

```python
from communitynet import GraphNet, MLP, CommunityNet

# Example numbers (arbitrary)
num_node_features = 4
num_edge_features = 2

base_gnn = GraphNet(in_channels=num_node_features, out_channels=8,
                    num_edge_features=num_edge_features)
output_gnn = nn.Sequential(
  GraphNet(in_channels=8, out_channels=4, num_edge_features=num_edge_features),
  MLP(in_channels=4, out_channels=1)
)
community_net = CommunityNet(base_gnn, output_gnn, num_communities=3)
```

The `CommunityNet` class itself derives from `torch.nn.Module`, so it can be trained like any other PyTorch model.

## API

### GraphNet

**`GraphNet(in_channels : int, out_channels : int, num_edge_features : int, hidden_channels : list = [], use_pooling : bool = False, dropout_prob : float = 0.0, global_pooling : str = "mean", activation : torch.nn.Module = None, edge_nn_kwargs : dict = {})`**

PyTorch module that uses `NNConv` (an edge-conditioned convolutional operator) as a filter and global pooling to convert a graph into a vector embedding.

**Parameters:**

TODO

### Multi-layer Perceptron (MLP)

PyTorch module that ...

### CommunityNet

PyTorch module that ...

<!--
Helpers for creating datasets (if each graph has same # of nodes, diff # of nodes, etc.)
-->
