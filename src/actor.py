import torch
from torch import tanh
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Sequential, LayerNorm


class Actor(Module):
    def __init__(self, state_dim, action_dim, action_scale, layer_sizes, epsilon=0.003):
        super().__init__()
        self.action_scale = action_scale
        layer_sizes = [state_dim] + layer_sizes

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers += [
                Linear(layer_sizes[i], layer_sizes[i + 1]),
                ReLU(),
            ]

        # Initialization of last layer
        last = Linear(layer_sizes[-1], action_dim)
        torch.nn.init.uniform_(last.weight, a=-epsilon, b=epsilon)  # Due to tanh activation and vanishing gradients
        layers.append(last)

        self.net = Sequential(*layers)

    def forward(self, state):
        action = self.net(state)
        action = self.action_scale * tanh(action)

        return action
