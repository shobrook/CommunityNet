# CommunityNet

`CommunityNet` is an ensemble Graph Neural Network (GNN) for graph datasets with community structure.<!--Why it's useful? Outperforms benchmarks? Classification only? Mention 'hierarchical'.-->

<img src="demo.png" />

## Installation

You can download `CommunityNet` from PyPi:

```bash
$ pip install communitynet
```

## Usage

Each graph you submit to `CommunityNet` must be an instance of `torch_geometric.data.Data` with an additional `communities` attribute. `data.communities` should hold a list of communities, where each community is a list of node indices.<!--Each graph in your dataset must have the same number of communities.-->



<!--
Walk through a classification example

`CommunityNet` is derives from `torch.nn.Module`, so it can be trained like any other PyTorch model.

Helpers for creating datasets (if each graph has same # of nodes, diff # of nodes, etc.)
-->
