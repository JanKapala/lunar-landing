import pickle
from copy import deepcopy, copy

import torch
from torch.nn import MSELoss, DataParallel
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.optim import SGD, Adam

from src.actor import Actor
from src.critic import Critic
from src.replay_buffer import ReplayBuffer
from src.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess


def negative_mean_loss_function(x):
    return -torch.mean(x)


class Agent:
    def __init__(
            self,
            device="cpu",
            state_dim=8,
            action_dim=2,
            action_high=None,
            actor_layer_sizes=None,
            critic_layer_sizes=None,
            replay_buffer_max_size=1e6,
            batch_size=128,
            γ=0.995,
            μ_θ_α=1e-4,
            Q_Φ_α=1e-3,
            ρ=0.95,
            exploration=True,
            noise_sigma=0.2,
            train_after=0,
            learning_freq=1,
            train_steps_per_update=1,
            writer=None,
    ):
        self.device = device

        # Actor
        self.μ_θ = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            action_scale=torch.Tensor(action_high).to(self.device),
            layer_sizes=actor_layer_sizes
        ).to(device)

        self.μ_θ_ℒ_function = negative_mean_loss_function  # Negative because gradient ascent
        self.μ_θ_α = μ_θ_α
        self.μ_θ_optimizer = Adam(self.μ_θ.parameters(), μ_θ_α)

        # Critic
        self.Q_Φ = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_sizes=critic_layer_sizes
        ).to(device)

        self.Q_Φ_ℒ_function = MSELoss()
        self.Q_Φ_α = Q_Φ_α
        self.Q_Φ_optimizer = Adam(self.Q_Φ.parameters(), Q_Φ_α)

        # Target networks
        self.ρ = ρ
        self.μ_θ_targ = deepcopy(self.μ_θ)
        self.Q_Φ_targ = deepcopy(self.Q_Φ)
        self.Q_Φ_targ.eval()
        self.μ_θ_targ.eval()

        # Replay Buffer
        self._batch_size = batch_size
        self.Ɗ = ReplayBuffer(
            max_size=replay_buffer_max_size,
            batch_size=batch_size
        )

        # Ornstein Uhlenbeck Process
        self.exploration = exploration
        self._noise_sigma = noise_sigma
        self._ouprocess = OrnsteinUhlenbeckProcess(
            action_dim=action_dim,
            sigma=noise_sigma
        )

        # Other hyper-parameters
        self.γ = γ

        self.train_after = train_after
        self.learning_freq = learning_freq
        self.train_steps_per_update = train_steps_per_update

        # Auxiliary variables
        self._last_S = None
        self._last_A = None

        self.steps_counter = 0
        self.episodes_counter = 0

        self.writer = writer

        self.returns = []
        self._last_return = 0

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self._batch_size = new_batch_size
        self.Ɗ.batch_size = new_batch_size

    @property
    def noise_sigma(self):
        return self._noise_sigma

    @noise_sigma.setter
    def noise_sigma(self, new_value):
        self._noise_sigma = new_value
        self._ouprocess.sigma = new_value

    def act(self, S):
        self._last_S = S

        S = self._prepare_state(S)
        # self.μ_θ.eval()
        with torch.no_grad():
            A = self.μ_θ(S)
            if self.exploration:
                A = self._add_noise(A)
            A = self._prepare_action(A)

        # self.μ_θ.train()

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

        # Critic loss calculation
        with torch.no_grad():
            y = R + self.γ * (1 - d) * self.Q_Φ_targ(S_prim, self.μ_θ_targ(S_prim))
        y_pred = self.Q_Φ(S, A)
        Q_Φ_ℒ = self.Q_Φ_ℒ_function(y_pred, y)

        # Weights update
        self.Q_Φ_optimizer.zero_grad()
        Q_Φ_ℒ.backward()
        # clip_grad_value_(self.Q_Φ.parameters(), clip_value=0.5)
        # clip_grad_norm_(self.Q_Φ.parameters(), max_norm=1.0, norm_type=2.0)
        self.Q_Φ_optimizer.step()

    def _actor_train_step(self, batch):
        S, _, _, _, _ = batch

        # Actor loss calculation
        A = self.μ_θ(S)
        action_value = self.Q_Φ(S, A)
        μ_θ_ℒ = self.μ_θ_ℒ_function(action_value)

        # Actor weights update
        self.μ_θ_optimizer.zero_grad()
        μ_θ_ℒ.backward()
        # clip_grad_value_(self.μ_θ.parameters(), clip_value=0.5)
        # clip_grad_norm_(self.μ_θ.parameters(), max_norm=1.0, norm_type=2.0)
        self.μ_θ_optimizer.step()

    def _target_nets_train_step(self):
        with torch.no_grad():
            for Φ, Φ_targ in zip(self.Q_Φ.parameters(), self.Q_Φ_targ.parameters()):
                Φ_targ.data.copy_(self.ρ * Φ_targ.data + (1 - self.ρ) * Φ.data)
            for θ, θ_targ in zip(self.μ_θ.parameters(), self.μ_θ_targ.parameters()):
                θ_targ.data.copy_(self.ρ * θ_targ.data + (1 - self.ρ) * θ.data)

    def _prepare_state(self, S):
        state_batch = torch.Tensor(S).unsqueeze(0).to(self.device)
        return state_batch

    def _prepare_action(self, A):
        A = A.squeeze(0).cpu().numpy()
        return A

    def _add_noise(self, action):
        noise = torch.Tensor(self._ouprocess.sample()).to(self.device)
        noised_action = action + noise
        return noised_action

    def to(self, device):
        writer = self.writer
        self.writer = None

        new_agent = deepcopy(self)
        self.writer = writer
        new_agent.writer = writer

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

        new_agent.μ_θ = new_agent.μ_θ.to(new_agent.device)
        new_agent.Q_Φ = new_agent.Q_Φ.to(new_agent.device)
        new_agent.μ_θ_targ = new_agent.μ_θ_targ.to(new_agent.device)
        new_agent.Q_Φ_targ = new_agent.Q_Φ_targ.to(new_agent.device)

        # Optimizer has to be instantiated after moving nets to selected device
        new_agent.μ_θ_optimizer = Adam(new_agent.μ_θ.parameters(), new_agent.μ_θ_α)
        new_agent.Q_Φ_optimizer = Adam(new_agent.Q_Φ.parameters(), new_agent.Q_Φ_α)

        new_agent.μ_θ.action_scale = new_agent.μ_θ.action_scale.to(new_agent.device)
        new_agent.μ_θ_targ.action_scale = new_agent.μ_θ_targ.action_scale.to(new_agent.device)

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

    def save(self, file_path):
        writer = self.writer
        self.writer = None
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        self.writer = writer
        print("Agent saved successfully! agent.writer object can't be saved so"
              " this filed has been set to `None`")

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
