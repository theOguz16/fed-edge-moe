import time

import ml.network.m13h_split_train_coordinator as base


original_prepare = base.prepare_split
original_gradient = base.compute_gradient


def timed_prepare():
    start = time.perf_counter()
    result = original_prepare()

    print(
        "[M13K] prepare:",
        f"{time.perf_counter() - start:.3f}s",
    )

    return result


def timed_gradient():
    start = time.perf_counter()
    result = original_gradient()

    print(
        "[M13K] server backward:",
        f"{time.perf_counter() - start:.3f}s",
    )

    return result


base.prepare_split = timed_prepare
base.compute_gradient = timed_gradient


if __name__ == "__main__":
    total_start = time.perf_counter()

    base.main()

    print(
        "[M13K] TOTAL E2E:",
        f"{time.perf_counter() - total_start:.3f}s",
    )
