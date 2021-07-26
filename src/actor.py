import torch
from torch import tanh
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Sequential


class Actor(Module):
    def __init__(self, state_dim=8, action_dim=2, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [128, 64, 32, 16]
        layer_sizes = [state_dim] + layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers += [
                Linear(layer_sizes[i], layer_sizes[i + 1]),
                ReLU(),
                BatchNorm1d(layer_sizes[i + 1]),
            ]
        last = Linear(layer_sizes[-1], action_dim)
        # torch.nn.init.uniform_(last.weight, a=-0.004, b=0.004)  # Due to tanh activation and vanishing gradients
        layers.append(last)

        # Experimental initialization
        for layer in layers:
            if isinstance(layer, Linear):
                # torch.nn.init.uniform_(layer.weight, a=-0.004, b=0.004)
                torch.nn.init.xavier_uniform_(layer.weight)


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
        action = tanh(action) # 2*tanh(action) ?
        return action
