import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Sequential, Identity


class Critic(Module):
    def __init__(self, state_dim, action_dim, layer_sizes, epsilon=0.003):
        super().__init__()

        # layer_sizes = [state_dim + action_dim] + layer_sizes
        # layers = []
        # for i in range(len(layer_sizes) - 1):
        #     layers += [
        #         Linear(layer_sizes[i], layer_sizes[i + 1]),
        #         BatchNorm1d(layer_sizes[i + 1]),
        #         ReLU(),
        #     ]
        # layers.append(Linear(layer_sizes[-1], 1))
        #
        # # Experimental initialization
        # for layer in layers:
        #     if isinstance(layer, Linear):
        #         # torch.nn.init.uniform_(layer.weight, a=-0.004, b=0.004)
        #         torch.nn.init.xavier_uniform_(layer.weight)
        #
        # self.net = Sequential(*layers)

        self.l_s = Linear(state_dim, layer_sizes[0])
        torch.nn.init.xavier_uniform_(self.l_s.weight)
        # self.l_s_bn = BatchNorm1d(layer_sizes[0])
        self.l_s_bn = Identity()

        self.l_a = Linear(action_dim, layer_sizes[0])
        torch.nn.init.xavier_uniform_(self.l_a.weight)
        # self.l_a_bn = BatchNorm1d(layer_sizes[0])
        self.l_a_bn = Identity()

        self.l_c = Linear(2*layer_sizes[0], layer_sizes[1])
        torch.nn.init.xavier_uniform_(self.l_c.weight)
        # self.l_c_bn = BatchNorm1d(layer_sizes[1])
        self.l_c_bn = Identity()

        self.l_output = Linear(layer_sizes[1], 1)
        torch.nn.init.uniform_(self.l_c.weight, a=-epsilon, b=epsilon)

    def forward(self, state, action):
        s_out = F.relu(self.l_s_bn(self.l_s(state)))
        a_out = F.relu(self.l_a_bn(self.l_a(action)))
        out = torch.cat([s_out, a_out], dim=1)
        out = F.relu(self.l_c_bn(self.l_c(out)))
        action_value = self.l_output(out)

        return action_value
