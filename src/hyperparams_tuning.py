from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from definitions import LOG_DIR
from src.agent_trainable import AgentTrainable

if __name__ == "__main__":
    CPU_N = 12
    GPU_N = 1

    PERC_SYSTEM_LOAD = 0.6

    MAX_CPU_N = PERC_SYSTEM_LOAD*CPU_N
    MAX_GPU_N = PERC_SYSTEM_LOAD*GPU_N

    TRIALS_N = 8

    cpu_per_trial = MAX_CPU_N/TRIALS_N
    # cpu_per_trial = 0.1
    gpu_per_trial = MAX_GPU_N/TRIALS_N

    config = {
        "batch_size": tune.choice([2 ** x for x in range(2, 12)]),
        "learning_freq": tune.choice([2 ** x for x in range(2, 9)]),
        "γ": tune.uniform(1e-2, 1),
        "μ_θ_α": tune.loguniform(1e-6, 1e-1),
        "Q_Φ_α": tune.loguniform(1e-6, 1e-1),
        "ρ": tune.loguniform(0.5, 0.95),
        "noise_scale": tune.uniform(0, 1),
        "train_steps_per_update": tune.choice([2 ** x for x in range(0, 6)]),
    }

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=4,
        hyperparam_mutations=config
    )

    analysis = tune.run(
        AgentTrainable,
        config=config,
        scheduler=pbt,
        metric="mean_return",
        mode="max",
        fail_fast=True,
        stop={"training_iteration": 3000, "mean_return": 100},
        num_samples=TRIALS_N,
        resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial},
        local_dir=LOG_DIR,
        # verbose=False
        reuse_actors=True
    )

    print("Best config: ", analysis.get_best_config(metric="mean_return", mode="max"))

    df = analysis.dataframe()
    print(df)
