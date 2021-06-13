# Third Party
import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, \
                               global_max_pool, TopKPooling

# Local Modules
from .mlp import MLP

GLOBAL_POOLING_LAYER_LOOKUP = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "max": global_max_pool,
    # "sort": global_sort_pool,
    # "attention":
}

class GraphNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_edge_features,
        hidden_channels=[],
        use_pooling=False,
        dropout_prob=0.0,
        global_pooling="mean",
        activation=None,
        edge_nn_kwargs={}
    ):
        super(GraphNet, self).__init__()

        self.conv_activation = activation if activation else lambda x: x
        self.use_pooling = use_pooling
        self.global_pooling_layer = GLOBAL_POOLING_LAYER_LOOKUP[global_pooling]
        self.dropout = nn.Dropout(p=dropout_prob)

        all_channels = [in_channels] + hidden_channels + [out_channels]
        self.layer_stack = nn.ModuleList()
        for index in range(len(all_channels) - 1):
            in_channels = all_channels[index]
            out_channels = all_channels[index + 1]

            edge_net = MLP(**{
                **{
                    "in_channels": num_edge_feats,
                    "out_channels": in_channels * out_channels
                },
                **edge_nn_kwargs
            })
            conv_layer = NNConv(
                in_channels,
                out_channels,
                nn=edge_net,
                aggr="mean"
            )
            self.layer_stack.append(conv_layer)

            if not use_pooling:
                continue

            self.layer_stack.append(TopKPooling(
                in_channels=out_channels,
                ratio=0.8
            ))

    def reset_parameters(self):
        for layer in self.layer_stack:
            layer.reset_parameters()

        self.dropout.reset_parameters()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        edge_attr, globals, batch = graph.edge_attr, graph.globals, graph.batch

        for index, layer in enumerate(self.layer_stack):
            if index % 2 and self.use_pooling: # Pooling layer
                x, edge_index, edge_attr, _, _ = layer(
                    x,
                    edge_index,
                    edge_attr,
                    # batch=batch
                )
            else: # Convolutional layer
                x = self.conv_activation(layer(x, edge_index, edge_attr))
                x = self.dropout(x)

        # Global pooling layer
        # batch = torch.tensor([0] * x.shape[0], dtype=torch.long).t()
        x = self.global_pooling_layer(x, batch).reshape(-1)

        return x
