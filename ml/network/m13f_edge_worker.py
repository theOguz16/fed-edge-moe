import argparse
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ml.model.expert import ExpertMLP


WORK = Path(
    "checkpoints/m13f_worker"
)

WORK.mkdir(
    parents=True,
    exist_ok=True,
)


def download(url, path):
    with urllib.request.urlopen(
        url,
        timeout=120,
    ) as response:
        data = response.read()

    path.write_bytes(data)

    return len(data)


def upload(url, path):
    data = path.read_bytes()

    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type":
                "application/octet-stream",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        return response.read().decode()


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
    print("FedEdgeMoE - M13F Edge Worker")
    print("Device:", device)
    print()

    job_path = WORK / "job.safetensors"
    expert_path = WORK / "expert.safetensors"
    result_path = WORK / "result.safetensors"

    job_bytes = download(
        server + "/job",
        job_path,
    )

    expert_bytes = download(
        server + "/expert",
        expert_path,
    )

    job = load_file(
        str(job_path)
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
    expert.eval()

    expert_input = (
        job["expert_input"]
        .to(device)
    )

    with torch.no_grad():
        output = expert(
            expert_input
        )

    save_file(
        {
            "expert_output":
                output
                .cpu()
                .contiguous(),
        },
        str(result_path),
    )

    response = upload(
        server + "/result",
        result_path,
    )

    print(
        "Job bytes:",
        job_bytes,
    )

    print(
        "Expert shard bytes:",
        expert_bytes,
    )

    print(
        "Result bytes:",
        result_path.stat().st_size,
    )

    print(
        "Server response:",
        response,
    )

    print(
        "EDGE FULL MODEL LOADED:",
        False,
    )


if __name__ == "__main__":
    main()
