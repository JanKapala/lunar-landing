import os
import pickle
import logging

import gym
import torch

from ray import tune
from torch.optim import SGD

from src.agent import Agent

EPISODES_N = 10000
LAST_EPISODES_FACTOR = 0.1


class AgentTrainable(tune.Trainable):
    def setup(self, config):
        # Instantiate environment and agent
        self.env = gym.make("LunarLanderContinuous-v2")

        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]

        self.agent = Agent(
            device="cpu",
            state_dim=state_dim,
            action_dim=action_dim,
            actor_layer_sizes=[32, 32],
            critic_layer_sizes=[32, 32],
            replay_buffer_max_size=1000,
            batch_size=config["batch_size"],
            learning_freq=config["learning_freq"],
            γ=config["γ"],
            μ_θ_α=config["μ_θ_α"],
            Q_Φ_α=config["Q_Φ_α"],
            ρ=config["ρ"],
            noise_scale=config["_noise_sigma"],
            train_after=1,
            exploration=True,
            train_steps_per_update=config["train_steps_per_update"],
        )

        # self.agent = self.agent.to("cpu")

    def step(self):
        d = False
        for episode_i in range(EPISODES_N):
            # logging.warning(f"episode_i: {episode_i}")
            S = self.env.reset()
            while not d:
                A = self.agent.act(S)
                S_prim, R, d, _ = self.env.step(A)
                self.agent.observe(R, S_prim, d)
                S = S_prim

        last_x_episodes = int(LAST_EPISODES_FACTOR * EPISODES_N)
        mean_return = torch.mean(torch.Tensor(self.agent.returns[-last_x_episodes:])).item()

        return {"mean_return": mean_return}

    def cleanup(self):
        self.env.close()

    def reset_config(self, new_config):
        self.agent.batch_size = new_config["batch_size"]
        self.agent.Ɗ.batch_size = new_config["batch_size"]
        self.agent.learning_freq = new_config["learning_freq"]
        self.agent.γ = new_config["γ"]
        self.agent.μ_θ_α = new_config["μ_θ_α"]
        self.agent.μ_θ_optimizer = SGD(self.agent.μ_θ.parameters(), self.agent.μ_θ_α)
        self.agent.Q_Φ_α = new_config["Q_Φ_α"]
        self.agent.Q_Φ_optimizer = SGD(self.agent.Q_Φ.parameters(), self.agent.Q_Φ_α)
        self.agent.ρ = new_config["ρ"]
        self.agent.noise_sigma = new_config["noise_sigma"]
        self.agent.train_steps_per_update = new_config["train_steps_per_update"]

        return True

    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        with open(path, 'wb') as file:
            pickle.dump(self.agent, file)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        with open(path, 'rb') as file:
            self.agent = pickle.load(file)
            self.agent = self.agent.to("cpu")
