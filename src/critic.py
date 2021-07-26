import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Sequential


class Critic(Module):
    def __init__(self, state_dim=8, action_dim=2, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [128, 64, 32, 16]
        layer_sizes = [state_dim + action_dim] + layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers += [
                Linear(layer_sizes[i], layer_sizes[i + 1]),
                ReLU(),
                BatchNorm1d(layer_sizes[i + 1]),

            ]
        layers.append(Linear(layer_sizes[-1], 1))

        # Experimental initialization
        for layer in layers:
            if isinstance(layer, Linear):
                # torch.nn.init.uniform_(layer.weight, a=-0.004, b=0.004)
                torch.nn.init.xavier_uniform_(layer.weight)

        self.net = Sequential(*layers)

        # self.l1_1 = Linear(state_dim, layer_sizes[0])
        # self.l1_1_bn = BatchNorm1d(layer_sizes[0])
        #
        # self.l1_2 = Linear(layer_sizes[0], layer_sizes[1])
        # self.l1_2_bn = BatchNorm1d(layer_sizes[1])
        #
        # self.l2_1 = Linear(action_dim, layer_sizes[1])
        # self.l2_1_bn = BatchNorm1d(layer_sizes[1])
        #
        # self.l_output = Linear(layer_sizes[1], 1)
        #
        # torch.nn.init.uniform_(self.l1_2.weight, a=-0.002, b=0.002)
        # torch.nn.init.uniform_(self.l_output.weight, a=-0.004, b=0.004)

    def forward(self, state, action):
        inp = torch.cat([state, action], dim=1)
        action_value = self.net(inp)

        # state_x = F.relu(self.l1_1_bn(self.l1_1(state)))
        # state_x = F.relu(self.l1_2_bn(self.l1_2(state_x)))
        #
        # action_x = F.relu(self.l2_1_bn(self.l2_1(action)))
        #
        # action_value = self.l_output(state_x + action_x)

        return action_value
