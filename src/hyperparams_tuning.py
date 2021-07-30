import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from definitions import LOG_DIR
from src.agent_trainable import AgentTrainable

# Ray Tune verbosity modes
SILENT = 0
ONLY_STATUS_UPDATES = 1
STATUS_AND_BRIEF_TRIAL_RESULTS = 2
STATUS_AND_DETAILED_TRIAL_RESULTS = 3

if __name__ == "__main__":
    CPU_N = 12
    GPU_N = 1

    PERC_SYSTEM_LOAD = 0.8

    MAX_CPU_N = PERC_SYSTEM_LOAD*CPU_N
    MAX_GPU_N = PERC_SYSTEM_LOAD*GPU_N

    TRIALS_N = 1

    cpu_per_trial = MAX_CPU_N/TRIALS_N
    if cpu_per_trial > 1:
        cpu_per_trial = int(np.floor(cpu_per_trial))

    gpu_per_trial = MAX_GPU_N/TRIALS_N
    if gpu_per_trial > 1:
        gpu_per_trial = int(np.floor(gpu_per_trial))

    cpu_per_trial = 1
    gpu_per_trial = 0

    print(f"""
    cpu_per_trial: {cpu_per_trial},
    gpu_per_trial: {gpu_per_trial},
    """)

    config = {
        "replay_buffer_max_size": tune.grid_search([10 ** x for x in range(3, 7)]),
        "batch_size": tune.grid_search([2 ** x for x in range(4, 10)]),
        "learning_freq": tune.grid_search([2 ** x for x in range(0, 5)]),
        "γ": tune.uniform(0.9, 1),
        "μ_θ_α": tune.loguniform(1e-6, 1e-1),
        "Q_Φ_α": tune.loguniform(1e-6, 1e-1),
        "ρ": tune.loguniform(0.5, 0.95),
        "noise_sigma": tune.uniform(0, 1),
    }

    hyperparam_mutations = {
        "replay_buffer_max_size": tune.choice([10 ** x for x in range(3, 7)]),
        "batch_size": tune.choice([2 ** x for x in range(4, 10)]),
        "learning_freq": tune.choice([2 ** x for x in range(0, 5)]),
        "γ": tune.uniform(0.9, 1),
        "μ_θ_α": tune.loguniform(1e-6, 1e-1),
        "Q_Φ_α": tune.loguniform(1e-6, 1e-1),
        "ρ": tune.loguniform(0.5, 0.95),
        "noise_sigma": tune.uniform(0, 1),
    }

    # Utility function
    def clip_limits(config, key, lower, upper):
        if config[key] < lower:
            config[key] = lower

        if config[key] > upper:
            config[key] = upper

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        clip_limits(config, "replay_buffer_max_size", 10**3, 10**6)
        clip_limits(config, "batch_size", 16, 512)
        clip_limits(config, "learning_freq", 1, 16)
        clip_limits(config, "γ", 0.9, 1)
        clip_limits(config, "μ_θ_α", 1e-6, 1e-1)
        clip_limits(config, "Q_Φ_α", 1e-6, 1e-1)
        clip_limits(config, "ρ", 0.5, 0.95)
        clip_limits(config, "noise_sigma", 0, 1)

        return config

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=1,
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
        resample_probability=0.25,
        synch=True
    )

    analysis = tune.run(
        AgentTrainable,
        config=config,
        scheduler=pbt,
        metric="mean_return",
        mode="max",
        fail_fast=True,
        stop={"training_iteration": 400, "mean_return": 200},
        num_samples=TRIALS_N,
        resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial},
        local_dir=LOG_DIR,
        verbose=ONLY_STATUS_UPDATES,
        reuse_actors=True,
        queue_trials=True,
    )

    print("Best config: ", analysis.get_best_config(metric="mean_return", mode="max"))

    df = analysis.dataframe()
    print(df)
