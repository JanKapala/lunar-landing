import gym
from gym.logger import ERROR

from definitions import LOG_DIR
from src.agent import Agent
from src.utils import timeit


def simulate(env, agent, steps, render=True):
    S = env.reset()
    for i in range(steps):
        if render:
            timeit(env.render)()

        A = timeit(agent.act)(S)

        S_prim, R, d, _ = timeit(env.step)(A)

        timeit(agent.observe)(R, S_prim, d)

        S = S_prim

        if d:
            S = env.reset()

    env.close()


if __name__ == "__main__":
    gym.logger.set_level(ERROR)

    env = gym.make("LunarLanderContinuous-v2")

    agent = Agent(
        device="cpu",
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        actor_layer_sizes=[64, 32, 16],
        critic_layer_sizes=[64, 32, 16],
        replay_buffer_max_size=10000,
        batch_size=64,
        learning_freq=64,
        γ=0.99,
        μ_θ_α=10e-6,
        Q_Φ_α=10e-6,
        ρ=0.95,
        noise_scale=0.1,
        train_after=64,
        exploration=True,
        writer=None,
        train_steps_per_update=32
    )
    agent = agent.to("cuda")
    simulate(env, agent, steps=10000, render=False)
    simulate(env, agent, steps=1000, render=True)
    print(LOG_DIR)
