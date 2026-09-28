import time

import ml.network.m12m_worker as proto


base = proto.base


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
        f"[M12N] worker eval "
        f"#{eval_count}: "
        f"{seconds:.3f}s"
    )

    return result


original_download = base.download_file


def timed_download(url, path):
    start = time.perf_counter()

    result = original_download(
        url,
        path,
    )

    total = time.perf_counter() - start

    print(
        f"[M12N] download total: "
        f"{total:.3f}s"
    )

    return result


original_post = base.post_bytes


def timed_post(
    url,
    payload,
    content_type,
):
    start = time.perf_counter()

    result = original_post(
        url,
        payload,
        content_type,
    )

    total = time.perf_counter() - start

    print(
        f"[M12N] POST "
        f"{url.split('/')[-1]}: "
        f"{total:.3f}s"
    )

    return result


base.evaluate_all = timed_eval
base.download_file = timed_download
base.post_bytes = timed_post


if __name__ == "__main__":
    base.main()
