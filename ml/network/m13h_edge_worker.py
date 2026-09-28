import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import (
    load_file,
    save_file,
)

from ml.model.expert import ExpertMLP


WORK = Path(
    "checkpoints/m13h_worker"
)

WORK.mkdir(
    parents=True,
    exist_ok=True,
)

LR = 5e-4


def download(url, path):
    with urllib.request.urlopen(
        url,
        timeout=120,
    ) as r:
        data = r.read()

    path.write_bytes(data)

    return len(data)


def post(url, path):
    data = path.read_bytes()

    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type":
                "application/octet-stream",
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=120,
    ) as r:
        return r.read()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server",
        required=True,
    )

    args = parser.parse_args()

    server = args.server.rstrip("/")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("FedEdgeMoE - M13H Worker")
    print("Device:", device)

    job_path = WORK / "job.safetensors"
    expert_path = WORK / "expert.safetensors"
    forward_path = WORK / "forward.safetensors"
    gradient_path = WORK / "gradient.safetensors"
    updated_path = WORK / "updated.safetensors"

    download(
        server + "/job",
        job_path,
    )

    download(
        server + "/expert",
        expert_path,
    )

    expert = ExpertMLP(
        d_model=128,
        hidden_dim=256,
    )

    expert.load_state_dict(
        load_file(
            str(expert_path)
        )
    )

    expert = expert.to(device)

    optimizer = torch.optim.AdamW(
        expert.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    expert_input = (
        load_file(
            str(job_path)
        )["expert_input"]
        .to(device)
    )

    output = expert(
        expert_input
    )

    save_file(
        {
            "expert_output":
                output.detach()
                .cpu()
                .contiguous(),
        },
        str(forward_path),
    )

    post(
        server + "/forward",
        forward_path,
    )

    while True:
        try:
            download(
                server + "/gradient",
                gradient_path,
            )
            break

        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise

            time.sleep(0.1)

    gradient = (
        load_file(
            str(gradient_path)
        )["gradient"]
        .to(device)
    )

    output.backward(
        gradient
    )

    optimizer.step()

    state = {
        key:
            value.detach()
            .cpu()
            .contiguous()

        for key, value
        in expert.state_dict().items()
    }

    save_file(
        state,
        str(updated_path),
    )

    post(
        server + "/updated",
        updated_path,
    )

    print(
        "Gradient bytes:",
        gradient_path.stat().st_size,
    )

    print(
        "Updated expert bytes:",
        updated_path.stat().st_size,
    )

    print(
        "FULL MODEL LOADED:",
        False,
    )

    print(
        "PHYSICAL EDGE TRAIN STEP COMPLETE"
    )


if __name__ == "__main__":
    main()
