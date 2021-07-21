import functools
from time import time

from torch.utils.tensorboard import SummaryWriter

from definitions import LOG_DIR

WRITER = SummaryWriter(log_dir=LOG_DIR)

def timeit(func):
    if "performance_measurements" not in globals():
        global performance_measurements
        performance_measurements = dict()

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start = time()
        ret_val = func(*args, **kwargs)
        end = time()
        elapsed = end - start

        key = f"{func.__name__}"
        if isinstance(performance_measurements.get(key), list):
            performance_measurements[key].append(elapsed)
        elif performance_measurements.get(key) is None:
            performance_measurements[key] = [elapsed]

        step = len(performance_measurements[key])
        WRITER.add_scalars("Performance", {key: elapsed}, step)
        WRITER.flush()

        return ret_val

    return newfunc


def show_times():
    if "performance_measurements" not in globals():
        global performance_measurements
        performance_measurements = dict()

    for key, value in performance_measurements.items():
        records = len(value)
        mean = sum(value) / records
        print(f"[{key}] Mean execution time: {mean}, records number: {records}")


def clear_times():
    if "performance_measurements" not in globals():
        global performance_measurements
    performance_measurements = dict()