import argparse
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
)


SEQ_LEN = 16
LEARNING_RATE = 5e-4

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
    "D2 (+3)",
    "D3 (+5)",
]

DOMAIN_STEPS = [1, 2, 3, 5]


def get_json(url):
    with urllib.request.urlopen(
        url,
        timeout=30,
    ) as response:
        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def download_file(url, path):
    start = time.perf_counter()

    with urllib.request.urlopen(
        url,
        timeout=120,
    ) as response:
        data = response.read()

    path.write_bytes(data)

    seconds = (
        time.perf_counter() - start
    )

    return len(data), seconds


def post_bytes(
    url,
    payload,
    content_type,
):
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type":
                content_type,
        },
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        body = response.read()

    seconds = (
        time.perf_counter() - start
    )

    return (
        json.loads(
            body.decode("utf-8")
        ),
        seconds,
    )


def build_model():
    return MiniMoELM(
        vocab_size=64,
        max_seq_len=32,
        d_model=128,
        num_heads=4,
        num_layers=3,
        num_experts=8,
        top_k=2,
        expert_hidden_dim=256,
        router_aux_loss_weight=0.01,
    )


def generate_domain_batch(
    domain,
    batch_size,
    seq_len,
    device,
):
    starts = torch.randint(
        0,
        16,
        (batch_size,),
        device=device,
    )

    positions = torch.arange(
        seq_len + 1,
        device=device,
    )

    base = domain * 16
    step_size = DOMAIN_STEPS[domain]

    sequence = (
        starts[:, None]
        + step_size * positions[None, :]
    ) % 16

    sequence = sequence + base

    return (
        sequence[:, :-1].long(),
        sequence[:, 1:].long(),
    )


def build_eval_set(
    domain,
    device,
):
    starts = torch.arange(
        16,
        device=device,
    )

    positions = torch.arange(
        SEQ_LEN + 1,
        device=device,
    )

    base = domain * 16
    step_size = DOMAIN_STEPS[domain]

    sequence = (
        starts[:, None]
        + step_size * positions[None, :]
    ) % 16

    sequence = sequence + base

    return (
        sequence[:, :-1].long(),
        sequence[:, 1:].long(),
    )


@torch.no_grad()
def evaluate_all(
    model,
    device,
):
    model.eval()

    results = []

    for domain in range(4):
        x, y = build_eval_set(
            domain,
            device,
        )

        output = model(
            input_ids=x,
            labels=y,
        )

        prediction = (
            output.logits.argmax(
                dim=-1
            )
        )

        accuracy = (
            prediction == y
        ).float().mean()

        results.append({
            "loss": float(
                output.lm_loss.detach().cpu()
            ),
            "accuracy": float(
                accuracy.detach().cpu()
            ),
        })

    return results


def clone_expert(expert):
    return {
        key: value.detach().cpu().clone()
        for key, value
        in expert.state_dict().items()
    }


def calculate_delta(before, after):
    return {
        key: (
            after[key] - before[key]
        ).contiguous()
        for key in before
    }


def delta_norm(delta):
    total = 0.0

    for tensor in delta.values():
        total += float(
            torch.sum(
                tensor.float() ** 2
            )
        )

    return total ** 0.5


def run_job(
    server,
    job,
    device,
):
    round_number = int(
        job["round"]
    )

    base_version = int(
        job["base_global_version"]
    )

    target_layer = int(
        job["target_layer"]
    )

    target_expert = int(
        job["target_expert"]
    )

    local_domain = int(
        job["local_domain"]
    )

    local_steps = int(
        job["local_steps"]
    )

    batch_size = int(
        job["batch_size"]
    )

    print()
    print("=" * 74)
    print(
        f"PHYSICAL ROUND {round_number}"
    )
    print(
        f"Global V{base_version}"
    )
    print(
        f"Target L{target_layer}-E{target_expert}"
    )
    print("=" * 74)

    round_dir = Path(
        "checkpoints/m8_worker"
    ) / f"round_{round_number:02d}"

    if round_dir.exists():
        shutil.rmtree(
            round_dir
        )

    round_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tar_path = (
        round_dir
        / "global_snapshot.tar.gz"
    )

    print(
        "Downloading global snapshot..."
    )

    download_bytes, download_seconds = (
        download_file(
            server + "/snapshot",
            tar_path,
        )
    )

    print(
        f"Downloaded {download_bytes:,} bytes "
        f"in {download_seconds:.3f}s"
    )

    with tarfile.open(
        tar_path,
        "r:gz",
    ) as tar:
        tar.extractall(
            round_dir,
            filter="data",
        )

    snapshot = (
        round_dir
        / "global_snapshot"
    )

    model = build_model()

    load_global_snapshot(
        model,
        snapshot,
    )

    model = model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    expert = (
        model
        .blocks[target_layer]
        .moe
        .experts[target_expert]
    )

    for parameter in expert.parameters():
        parameter.requires_grad = True

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    before_eval = evaluate_all(
        model,
        device,
    )

    print()
    print(
        "Before local training:"
    )

    for index, result in enumerate(
        before_eval
    ):
        print(
            f"{DOMAIN_NAMES[index]:<8} "
            f"acc={result['accuracy'] * 100:6.2f}%"
        )

    before = clone_expert(
        expert
    )

    torch.manual_seed(
        9000 + round_number
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    training_start = (
        time.perf_counter()
    )

    model.train()

    running_loss = 0.0

    for step in range(
        1,
        local_steps + 1,
    ):
        x, y = generate_domain_batch(
            local_domain,
            batch_size,
            SEQ_LEN,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            input_ids=x,
            labels=y,
        )

        output.lm_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += float(
            output.lm_loss.detach().cpu()
        )

        if step % 30 == 0:
            print(
                f"step {step:03d}/{local_steps} "
                f"avg_loss="
                f"{running_loss / 30:.6f}"
            )

            running_loss = 0.0

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    after_eval = evaluate_all(
        model,
        device,
    )

    print()
    print(
        "After local training:"
    )

    for index, result in enumerate(
        after_eval
    ):
        print(
            f"{DOMAIN_NAMES[index]:<8} "
            f"acc={result['accuracy'] * 100:6.2f}%"
        )

    after = clone_expert(
        expert
    )

    delta = calculate_delta(
        before,
        after,
    )

    delta_path = (
        round_dir
        / "expert_delta.safetensors"
    )

    save_file(
        delta,
        str(delta_path),
    )

    delta_bytes = (
        delta_path.stat().st_size
    )

    metadata = {
        "client_id":
            "client_B_physical_msi",

        "round":
            round_number,

        "domain":
            local_domain,

        "base_global_version":
            base_version,

        "layer":
            target_layer,

        "expert":
            target_expert,

        "local_steps":
            local_steps,

        "batch_size":
            batch_size,

        "local_tokens_seen":
            local_steps
            * batch_size
            * SEQ_LEN,

        "trainable_parameters":
            sum(
                p.numel()
                for p in trainable
            ),

        "training_seconds":
            training_seconds,

        "snapshot_download_bytes":
            download_bytes,

        "snapshot_download_seconds":
            download_seconds,

        "delta_l2_norm":
            delta_norm(delta),

        "delta_bytes":
            delta_bytes,

        "before":
            before_eval,

        "after":
            after_eval,

        "device":
            str(device),

        "gpu":
            torch.cuda.get_device_name(0),
    }

    metadata_bytes = json.dumps(
        metadata,
        indent=2,
    ).encode("utf-8")

    print()
    print(
        "Submitting versioned metadata..."
    )

    response, _ = post_bytes(
        server
        + "/upload/client_B/metadata",
        metadata_bytes,
        "application/json",
    )

    print(response)

    print(
        "Uploading expert delta..."
    )

    delta_payload = (
        delta_path.read_bytes()
    )

    response, upload_seconds = (
        post_bytes(
            server
            + "/upload/client_B/delta",
            delta_payload,
            "application/octet-stream",
        )
    )

    print(response)

    print(
        f"Delta upload time: "
        f"{upload_seconds:.3f}s"
    )

    print(
        f"Round {round_number} complete."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server",
        required=True,
    )

    args = parser.parse_args()

    server = (
        args.server.rstrip("/")
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        "FedEdgeMoE - M8 "
        "Physical Worker Loop"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Server:",
        server,
    )

    last_round = 0

    while True:
        try:
            job = get_json(
                server + "/job"
            )

        except Exception as exc:
            print(
                "Coordinator unavailable:",
                exc,
            )

            time.sleep(2)
            continue

        if job.get("complete"):
            print()
            print(
                "Coordinator reports "
                "all rounds complete."
            )
            break

        if not job.get("ready"):
            time.sleep(1)
            continue

        round_number = int(
            job["round"]
        )

        if round_number <= last_round:
            time.sleep(1)
            continue

        try:
            run_job(
                server,
                job,
                device,
            )

            last_round = (
                round_number
            )

        except urllib.error.HTTPError as exc:
            body = (
                exc.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

            print(
                "Coordinator rejected update:",
                exc.code,
                body,
            )

            time.sleep(2)

        except Exception as exc:
            print(
                "Worker error:",
                repr(exc),
            )

            raise


if __name__ == "__main__":
    main()
