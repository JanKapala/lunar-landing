import os
import pickle
import logging

import gym
import torch

from ray import tune
from torch.optim import SGD, Adam

from src.agent import Agent
from src.simulation import simulate

EPISODES_N = 10
MAX_EPISODE_STEPS = 1000
LAST_EPISODES_FACTOR = 1


class AgentTrainable(tune.Trainable):
    def setup(self, config):
        self.config = config

        # Instantiate environment and agent
        self.env = gym.make("LunarLanderContinuous-v2")

        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]
        action_high = self.env.action_space.high

        self.agent = Agent(
            device="cuda",
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            actor_layer_sizes=[256, 128],
            critic_layer_sizes=[256, 128],
            replay_buffer_max_size=config["replay_buffer_max_size"],
            batch_size=config["batch_size"],
            learning_freq=config["learning_freq"],
            γ=config["γ"],
            μ_θ_α=config["μ_θ_α"],
            Q_Φ_α=config["Q_Φ_α"],
            ρ=config["ρ"],
            exploration=True,
            noise_sigma=config["noise_sigma"],
            train_after=1,
            # train_steps_per_update=config["train_steps_per_update"],
            train_steps_per_update=config["learning_freq"],  # keep them synchronized to keep the same execution time
            writer=None,
        )

    def step(self):
        simulate(self.env, self.agent, episodes=EPISODES_N,
                 max_episode_steps=MAX_EPISODE_STEPS, render=False)

        last_n_episodes = int(LAST_EPISODES_FACTOR * EPISODES_N)
        mean_return = self.agent.evaluate(last_n_episodes)

        return {"mean_return": mean_return}

    def cleanup(self):
        self.env.close()

    # def reset_config(self, new_config):
    #     self.agent.replay_buffer_max_size = new_config["replay_buffer_max_size"]
    #     self.agent.batch_size = new_config["batch_size"]
    #     self.agent.learning_freq = new_config["learning_freq"]
    #     self.agent.γ = new_config["γ"]
    #     self.agent.μ_θ_α = new_config["μ_θ_α"]
    #     self.agent.Q_Φ_α = new_config["Q_Φ_α"]
    #     self.agent.ρ = new_config["ρ"]
    #     self.agent.noise_sigma = new_config["noise_sigma"]
    #     # self.agent.train_steps_per_update = new_config["train_steps_per_update"]
    #     self.agent.train_steps_per_update = new_config["learning_freq"],  # keep them synchronized to keep the same execution time
    #
    #     return True

    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        self.agent.save(path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "checkpoint")

        # PBT exploitation phase is realized via `load_checkpoint` method so we
        # want to load saved agent with its:
        #   - weights,
        #   - optimizers,
        #   - replay buffer,
        #   - etc
        # but we want to use current config's hyperparams rather than loaded agent
        # hyperparams, so we have to update them:

        self.agent = Agent.load(path)

        for hp_name, hp_value in self.config.items():
            setattr(self.agent, hp_name, hp_value)

        # train_steps_per_update and learning_freq should be
        # synchronized to keep the same execution time among trials
        self.agent.train_steps_per_update = self.config["learning_freq"]

