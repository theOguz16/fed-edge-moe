import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import generate_balanced_batch
from ml.federated.checkpoint_manager import load_global_snapshot
from ml.distributed.expert_dispatch import prepare_expert_job


HOST = "0.0.0.0"
PORT = 8770

CHECKPOINT = Path(
    "checkpoints/m10_candidates/global_step_0100"
)

EXPERT_SHARD = Path(
    "checkpoints/m13a/v0100_L0_E7/"
    "experts/layer_00/"
    "expert_07_v0100.safetensors"
)

WORK = Path("checkpoints/m13f")
WORK.mkdir(parents=True, exist_ok=True)

JOB_PATH = WORK / "job.safetensors"
RESULT_PATH = WORK / "result.safetensors"

RESULT_EVENT = threading.Event()

REFERENCE = None


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


def prepare():
    global REFERENCE

    torch.manual_seed(20260927)

    model = build_model()
    load_global_snapshot(
        model,
        CHECKPOINT,
    )
    model.eval()

    input_ids, _ = generate_balanced_batch(
        split="validation",
        batch_size=32,
        seq_len=16,
        device=torch.device("cpu"),
    )

    seq_len = input_ids.shape[1]

    positions = torch.arange(seq_len)

    x = (
        model.token_embedding(input_ids)
        + model.position_embedding(
            positions
        )[None, :, :]
    )

    block = model.blocks[0]

    mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    attn_input = block.attn_norm(x)

    attn_output, _ = block.attention(
        attn_input,
        attn_input,
        attn_input,
        attn_mask=mask,
        need_weights=False,
    )

    x = x + attn_output
    moe_input = block.moe_norm(x)

    route = block.moe.router(
        moe_input
    )

    job = prepare_expert_job(
        moe_input,
        route,
        7,
    )

    with torch.no_grad():
        REFERENCE = (
            block.moe.experts[7](
                job.expert_input
            )
            .cpu()
        )

    save_file(
        {
            "expert_input":
                job.expert_input
                .cpu()
                .contiguous(),
        },
        str(JOB_PATH),
    )

    return job.token_indices.numel()


class Handler(BaseHTTPRequestHandler):
    def log_message(
        self,
        fmt,
        *args,
    ):
        print(
            "[HTTP]",
            self.address_string(),
            fmt % args,
        )

    def send_file(self, path):
        data = path.read_bytes()

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/octet-stream",
        )
        self.send_header(
            "Content-Length",
            str(len(data)),
        )
        self.end_headers()

        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/job":
            self.send_file(JOB_PATH)
            return

        if self.path == "/expert":
            self.send_file(EXPERT_SHARD)
            return

        self.send_error(404)

    def do_POST(self):
        if self.path != "/result":
            self.send_error(404)
            return

        length = int(
            self.headers["Content-Length"]
        )

        payload = self.rfile.read(length)
        RESULT_PATH.write_bytes(payload)

        RESULT_EVENT.set()

        body = json.dumps({
            "status": "accepted",
            "bytes": len(payload),
        }).encode()

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()

        self.wfile.write(body)


def main():
    routed = prepare()

    print()
    print("FedEdgeMoE - M13F")
    print("Physical Split Expert Execution")
    print()
    print(
        "Coordinator:",
        f"{HOST}:{PORT}",
    )
    print(
        "Target: L0-E7"
    )
    print(
        "Routed tokens:",
        routed,
    )
    print(
        "Waiting for MSI..."
    )

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    RESULT_EVENT.wait(
        timeout=300
    )

    server.shutdown()

    if not RESULT_PATH.exists():
        raise RuntimeError(
            "No MSI result received"
        )

    result = load_file(
        str(RESULT_PATH)
    )["expert_output"]

    max_diff = (
        REFERENCE
        - result
    ).abs().max().item()

    print()
    print(
        "Job bytes:",
        JOB_PATH.stat().st_size,
    )

    print(
        "Expert shard bytes:",
        EXPERT_SHARD.stat().st_size,
    )

    print(
        "Result bytes:",
        RESULT_PATH.stat().st_size,
    )

    print(
        "Max difference:",
        max_diff,
    )

    print(
        "PHYSICAL SPLIT EXACT:",
        max_diff < 1e-6,
    )


if __name__ == "__main__":
    main()
