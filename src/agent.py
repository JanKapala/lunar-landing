import random
from copy import deepcopy, copy

import torch
from torch.nn import MSELoss, DataParallel
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.optim import SGD

from src.actor import Actor
from src.critic import Critic
from src.replay_buffer import ReplayBuffer

class Agent:
    def __init__(
            self,
            device="cpu",
            state_dim=8,
            action_dim=2,
            actor_layer_sizes=None,
            critic_layer_sizes=None,
            replay_buffer_max_size=1000,
            batch_size=64,
            learning_freq=1,
            γ=0.99,
            μ_θ_α=0.01,
            Q_Φ_α=0.01,
            ρ=0.95,
            noise_scale=0.1,
            train_after=0,
            exploration=True,
            writer=None,
            train_steps_per_update=1,
            action_low=None,
            action_high=None
    ):
        if action_high is None:
            action_high = [1, 1]
        if action_low is None:
            action_low = [-1, -1]
        if critic_layer_sizes is None:
            critic_layer_sizes = [64, 32, 16]
        if actor_layer_sizes is None:
            actor_layer_sizes = [64, 32, 16]
        self.device = device

        self.action_dim = action_dim
        self.μ_θ = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_sizes=actor_layer_sizes
        ).to(device)

        self.Q_Φ = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_sizes=critic_layer_sizes
        ).to(device)

        self.μ_θ_targ = deepcopy(self.μ_θ)
        self.Q_Φ_targ = deepcopy(self.Q_Φ)
        self.Q_Φ_targ.eval()
        self.μ_θ_targ.eval()

        self.batch_size = batch_size
        self.Ɗ = ReplayBuffer(
            max_size=replay_buffer_max_size,
            batch_size=batch_size
        )
        self.learning_freq = learning_freq
        self.γ = γ

        self.MSE = MSELoss()

        self.μ_θ_α = μ_θ_α
        self.μ_θ_optimizer = SGD(self.μ_θ.parameters(), μ_θ_α)

        self.Q_Φ_α = Q_Φ_α
        self.Q_Φ_optimizer = SGD(self.Q_Φ.parameters(), Q_Φ_α)

        self.ρ = ρ
        self.noise_scale = noise_scale

        self._last_S = None
        self._last_A = None

        self.steps_counter = 0
        self.episodes_counter = 0

        self.returns = []
        self._last_return = 0

        self.train_after = train_after
        self.exploration = exploration

        self.writer = writer

        self.train_steps_per_update = train_steps_per_update

        self.action_low = action_low
        self.action_high = action_high

    def act(self, S):
        self._last_S = S

        S = self._prepare_state(S)
        self.μ_θ.eval()
        with torch.no_grad():
            A = self.μ_θ(S)
            if self.exploration:
                A = self._ornstein_uhlenbeck_process(A)
            A = self._prepare_action(A)

        self._last_A = A

        return A

    def observe(self, R, S_prim, d):
        self._last_return += R

        if d:
            return_val = copy(self._last_return)
            self.returns.append(return_val)
            self._last_return = 0
            self._log()
            self.episodes_counter += 1

        S = self._last_S
        A = self._last_A

        self.Ɗ << (S, A, R, S_prim, d)

        self.steps_counter += 1

        if self._update_time():
            for _ in range(self.train_steps_per_update):
                self.train_step()

    def _update_time(self):
        flag = (
                self.steps_counter >= self.train_after
                and
                # self.steps_counter >= self.batch_size
                # and
                self.steps_counter % self.learning_freq == 0
        )
        return flag

    def train_step(self):
        batch = next(self.Ɗ)

        batch = tuple([t.to(self.device) for t in batch])

        self._critic_train_step(batch)
        self._actor_train_step(batch)
        self._target_nets_train_step()

    def _critic_train_step(self, batch):
        S, A, R, S_prim, d = batch

        self.μ_θ.eval()
        with torch.no_grad():
            y = R + self.γ * (1 - d) * self.Q_Φ_targ(S_prim, self.μ_θ_targ(S_prim))
        self.Q_Φ.train()
        Q_Φ_ℒ = self.MSE(self.Q_Φ(S, A), y)
        Q_Φ_ℒ.backward()
        clip_grad_value_(self.Q_Φ.parameters(), clip_value=0.5)
        clip_grad_norm_(self.Q_Φ.parameters(), max_norm=1.0, norm_type=2.0)
        self.Q_Φ_optimizer.step()

    def _actor_train_step(self, batch):
        S, A, R, S_prim, d = batch

        self.Q_Φ.eval()
        self.μ_θ.train()
        μ_θ_ℒ = -torch.mean(self.Q_Φ(S, self.μ_θ(S)))  # Minus because gradient ascent
        μ_θ_ℒ.backward()
        clip_grad_value_(self.μ_θ.parameters(), clip_value=0.5)
        clip_grad_norm_(self.μ_θ.parameters(), max_norm=1.0, norm_type=2.0)
        self.μ_θ_optimizer.step()

    def _target_nets_train_step(self):
        for Φ, Φ_targ in zip(self.Q_Φ.parameters(), self.Q_Φ_targ.parameters()):
            Φ_targ.data = self.ρ * Φ_targ.data + (1 - self.ρ) * Φ.data
        for θ, θ_targ in zip(self.μ_θ.parameters(), self.μ_θ_targ.parameters()):
            θ_targ.data = self.ρ * θ_targ.data + (1 - self.ρ) * θ.data

    def _prepare_state(self, S):
        state_batch = torch.Tensor(S).unsqueeze(0).to(self.device)
        return state_batch

    def _prepare_action(self, A):
        A = A.squeeze_(0).cpu().numpy()
        return A

    def _add_noise(self, action):
        action = action.cpu()
        low = torch.Tensor(self.action_low)
        high = torch.Tensor(self.action_high)
        the_range = high - low
        normal = torch.normal(0.5, 0.5 * 1 / 3, (2,))
        deviation = the_range * normal
        epsilon = deviation * self.noise_scale * (-1) ** random.randint(0, 1)
        noised_action = torch.max(torch.min(action + epsilon, high), low)

        noised_action = noised_action.to(self.device)

        return noised_action

    def _ornstein_uhlenbeck_process(self, action, mu=0, dt=0.1, std=0.2):
        """Ornstein–Uhlenbeck process"""

        action = action.cpu()

        dt_sqrt = torch.sqrt(torch.tensor(dt))
        normal = torch.normal(mean=torch.zeros(self.action_dim))
        noised_action = action + self.noise_scale * (mu - action) * dt + std * dt_sqrt * normal

        noised_action = noised_action.to(self.device)

        return noised_action

    def to(self, device):
        if self.writer is not None:
            raise Exception("It is impossible to copy agent with self.writer object. Set it to None and try again.")

        new_agent = deepcopy(self)

        new_agent.device = device

        if new_agent.device == "cuda":
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    new_agent.μ_θ = DataParallel(new_agent.μ_θ)
                    new_agent.Q_Φ = DataParallel(new_agent.Q_Φ)
            else:
                new_agent.device = "cpu"

        new_agent.μ_θ_targ = deepcopy(new_agent.μ_θ)
        new_agent.Q_Φ_targ = deepcopy(new_agent.Q_Φ)
        new_agent.μ_θ_targ.eval()
        new_agent.Q_Φ_targ.eval()

        new_agent.μ_θ.to(new_agent.device)
        new_agent.Q_Φ.to(new_agent.device)
        new_agent.μ_θ_targ.to(new_agent.device)
        new_agent.Q_Φ_targ.to(new_agent.device)

        # Optimizer has to be instantiated after moving nets to selected device
        new_agent.μ_θ_optimizer = SGD(new_agent.μ_θ.parameters(), new_agent.μ_θ_α)
        new_agent.Q_Φ_optimizer = SGD(new_agent.Q_Φ.parameters(), new_agent.Q_Φ_α)

        return new_agent

    def _log(self):
        if self.writer is None:
            return

        nets = {
            "μ_θ": self.μ_θ,
            "Q_Φ": self.Q_Φ,
            "μ_θ_targ": self.μ_θ_targ,
            "Q_Φ_targ": self.Q_Φ_targ,
        }
        for net_name, net in nets.items():
            for param_name, param in net.named_parameters():
                self.writer.add_histogram(f"{net_name} {param_name}", param.data, self.episodes_counter)

        self.writer.add_scalar("Return", self.returns[-1], self.episodes_counter)
        self.writer.add_scalar("Replay Buffer size", len(self.Ɗ.buffer), self.steps_counter)

        self.writer.flush()
