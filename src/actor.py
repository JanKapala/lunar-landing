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
                BatchNorm1d(layer_sizes[i + 1])
            ]
        layers.append(Linear(layer_sizes[-1], action_dim))

        self.net = Sequential(*layers)

    def forward(self, state):
        action = self.net(state)
        return action
