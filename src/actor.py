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
                # BatchNorm1d(layer_sizes[i + 1]),
                LayerNorm(layer_sizes[i + 1]),
                ReLU(),
            ]

        # Initialization of all layers except last
        for layer in layers:
            if isinstance(layer, Linear):
                # torch.nn.init.uniform_(layer.weight, a=-0.004, b=0.004)
                torch.nn.init.xavier_uniform_(layer.weight)

        # Initialization of last layer
        last = Linear(layer_sizes[-1], action_dim)
        torch.nn.init.uniform_(last.weight, a=-epsilon, b=epsilon)  # Due to tanh activation and vanishing gradients
        layers.append(last)

        self.net = Sequential(*layers)

        # l1 = Linear(state_dim, layer_sizes[0])
        # l2 = Linear(layer_sizes[0], layer_sizes[1])
        # l3 = Linear(layer_sizes[1], action_dim)
        #
        # self.net = Sequential(
        #     l1,
        #     BatchNorm1d(layer_sizes[0]),
        #     ReLU(),
        #     l2,
        #     BatchNorm1d(layer_sizes[1]),
        #     ReLU(),
        #     l3,
        #     BatchNorm1d(action_dim),
        #     ReLU(),
        # )
        #
        # torch.nn.init.uniform_(l2.weight, a=-0.002, b=0.002)
        # torch.nn.init.uniform_(l3.weight, a=-0.004, b=0.004)

    def forward(self, state):
        action = self.net(state)
        action = self.action_scale * tanh(action)

        return action
