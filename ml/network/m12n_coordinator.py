import time
from pathlib import Path

import ml.network.m12m_coordinator as proto


base = proto.base

base.ROOT = Path("checkpoints/m12n")
base.RESULT_PATH = Path(
    "results/m12n_physical_timing.json"
)


original_eval = base.evaluate_all
eval_count = 0


def timed_eval(model, device):
    global eval_count

    start = time.perf_counter()
    result = original_eval(
        model,
        device,
    )
    seconds = time.perf_counter() - start

    eval_count += 1

    print(
        f"[M12N] coordinator eval "
        f"#{eval_count}: "
        f"{seconds:.3f}s"
    )

    return result


original_tar = base.create_snapshot_tar


def timed_tar(*args, **kwargs):
    start = time.perf_counter()

    result = original_tar(
        *args,
        **kwargs,
    )

    seconds = time.perf_counter() - start

    print(
        f"[M12N] snapshot tar: "
        f"{seconds:.3f}s"
    )

    return result


original_train_a = base.train_client_a


def timed_train_a(*args, **kwargs):
    start = time.perf_counter()

    result = original_train_a(
        *args,
        **kwargs,
    )

    total = time.perf_counter() - start

    delta, meta = result

    optimizer_time = float(
        meta["training_seconds"]
    )

    print(
        f"[M12N] Client A total: "
        f"{total:.3f}s"
    )

    print(
        f"[M12N] Client A optimizer: "
        f"{optimizer_time:.3f}s"
    )

    print(
        f"[M12N] Client A overhead: "
        f"{total - optimizer_time:.3f}s"
    )

    return delta, meta


original_export = (
    base.export_global_snapshot
)


def timed_export(*args, **kwargs):
    start = time.perf_counter()

    result = original_export(
        *args,
        **kwargs,
    )

    seconds = time.perf_counter() - start

    print(
        f"[M12N] checkpoint export: "
        f"{seconds:.3f}s"
    )

    return result


base.evaluate_all = timed_eval
base.create_snapshot_tar = timed_tar
base.train_client_a = timed_train_a
base.export_global_snapshot = timed_export


if __name__ == "__main__":
    base.main()
