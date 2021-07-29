import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Sequential, Identity


class Critic(Module):
    def __init__(self, state_dim, action_dim, layer_sizes, epsilon=0.003):
        super().__init__()

        self.l_s = Linear(state_dim, layer_sizes[0])
        self.l_a = Linear(action_dim, layer_sizes[0])
        self.l_c = Linear(2*layer_sizes[0], layer_sizes[1])

        self.l_output = Linear(layer_sizes[1], 1)
        torch.nn.init.uniform_(self.l_c.weight, a=-epsilon, b=epsilon)

        # self.l1 = Linear(state_dim + action_dim, layer_sizes[0])
        # self.l2 = Linear(layer_sizes[0], layer_sizes[1])
        # self.l3 = Linear(layer_sizes[1], 1)
        # torch.nn.init.uniform_(self.l3.weight, a=-epsilon, b=epsilon)

    def forward(self, state, action):
        s_out = F.relu(self.l_s(state))
        a_out = F.relu(self.l_a(action))
        out = torch.cat([s_out, a_out], dim=1)
        out = F.relu(self.l_c(out))
        action_value = self.l_output(out)
        # x = torch.cat([state, action], dim=1)
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # action_value = self.l3(x)

        return action_value
