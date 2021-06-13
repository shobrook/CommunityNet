# Third Party
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=[],
        h_activation=None,
        out_activation=None
    ):
        super(MLP, self).__init__()

        self.h_activation = h_activation if h_activation else lambda x: x
        self.out_activation = out_activation if out_activation else lambda x: x

        all_channels = [in_channels] + hidden_channels + [out_channels]
        self.layers = nn.ModuleList([])
        for i in range(len(all_channels) - 1):
            self.layers.append(nn.Linear(all_channels[i], all_channels[i + 1]))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, input):
        for index, layer in enumerate(self.layers):
            input = layer(input) # QUESTION: Wrap this in self.h_activation?

            if index == len(self.layers) - 1:
                input = self.out_activation(input)
            else:
                input = self.h_activation(input)

        return input
