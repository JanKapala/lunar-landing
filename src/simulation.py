import gym
from gym.logger import ERROR
from tqdm.auto import trange

from definitions import LOG_DIR
from src.agent import Agent
from src.utils import timeit


def simulate(env, agent, episodes, render=True, max_episode_steps=None, progress_bar=True):
    if progress_bar:
        custom_range = trange
    else:
        custom_range = range
    for _ in custom_range(episodes):
        S = env.reset()
        episode_steps = 0
        while True:
            if render:
                env.render()

            A = agent.act(S)
            S_prim, R, d, _ = env.step(A)

            if max_episode_steps is not None and episode_steps >= max_episode_steps:
                d = True
            agent.observe(R, S_prim, d)
            S = S_prim
            episode_steps += 1
            if d:
                break
    env.close()
