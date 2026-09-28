import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot
from ml.distributed.expert_dispatch import (
    prepare_expert_job,
    merge_expert_result,
)


HOST = "0.0.0.0"
PORT = 8771

CHECKPOINT = (
    "checkpoints/m10_candidates/global_step_0100"
)

EXPERT_SHARD = (
    "checkpoints/m13a/v0100_L0_E7/"
    "experts/layer_00/"
    "expert_07_v0100.safetensors"
)

WORK = Path("checkpoints/m13h")
WORK.mkdir(parents=True, exist_ok=True)

JOB = WORK / "job.safetensors"
FORWARD = WORK / "forward.safetensors"
GRADIENT = WORK / "gradient.safetensors"
UPDATED = WORK / "updated.safetensors"

FORWARD_EVENT = threading.Event()
UPDATED_EVENT = threading.Event()

PHYSICAL_START = None
PHYSICAL_END = None
GRADIENT_READY = threading.Event()

LR = 5e-4

SERVER_MODEL = None
REFERENCE_AFTER = None
TARGET_JOB = None
PRE_MOE_H = None
MOE_INPUT = None
ROUTE = None
X = None
Y = None


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


def batch():
    pool = build_split_tensor(
        1,
        "train",
        16,
    )

    seq = pool[:32]

    x = seq[:, :-1]
    y = seq[:, 1:].clone()

    y[:, 0] = -100

    return x, y


def clone_state(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def reference_step(x, y):
    model = build_model()
    load_global_snapshot(
        model,
        CHECKPOINT,
    )

    for p in model.parameters():
        p.requires_grad = False

    expert = model.blocks[0].moe.experts[7]

    for p in expert.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        expert.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    opt.zero_grad(set_to_none=True)

    output = model(input_ids=x)

    loss = F.cross_entropy(
        output.logits.reshape(
            -1,
            64,
        ),
        y.reshape(-1),
    )

    loss.backward()
    opt.step()

    return (
        float(loss.detach()),
        clone_state(expert),
    )


def prepare_split():
    global SERVER_MODEL
    global REFERENCE_AFTER
    global TARGET_JOB
    global PRE_MOE_H
    global MOE_INPUT
    global ROUTE
    global X
    global Y

    torch.manual_seed(20260927)
    torch.set_num_threads(1)

    X, Y = batch()

    ref_loss, REFERENCE_AFTER = (
        reference_step(X, Y)
    )

    SERVER_MODEL = build_model()

    load_global_snapshot(
        SERVER_MODEL,
        CHECKPOINT,
    )

    for p in SERVER_MODEL.parameters():
        p.requires_grad = False

    seq_len = X.shape[1]

    positions = torch.arange(
        seq_len
    )

    h = (
        SERVER_MODEL.token_embedding(X)
        + SERVER_MODEL.position_embedding(
            positions
        )[None, :, :]
    )

    block = SERVER_MODEL.blocks[0]

    mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    attn_in = block.attn_norm(h)

    attn_out, _ = block.attention(
        attn_in,
        attn_in,
        attn_in,
        attn_mask=mask,
        need_weights=False,
    )

    PRE_MOE_H = h + attn_out

    MOE_INPUT = block.moe_norm(
        PRE_MOE_H
    )

    ROUTE = block.moe.router(
        MOE_INPUT
    )

    TARGET_JOB = prepare_expert_job(
        MOE_INPUT,
        ROUTE,
        7,
    )

    save_file(
        {
            "expert_input":
                TARGET_JOB
                .expert_input
                .contiguous(),
        },
        str(JOB),
    )

    return ref_loss


def compute_gradient():
    forward = load_file(
        str(FORWARD)
    )["expert_output"]

    boundary = (
        forward
        .clone()
        .detach()
        .requires_grad_(True)
    )

    block = SERVER_MODEL.blocks[0]

    flat = torch.zeros_like(
        MOE_INPUT.reshape(-1, 128)
    )

    for expert_id, expert in enumerate(
        block.moe.experts
    ):
        job = prepare_expert_job(
            MOE_INPUT,
            ROUTE,
            expert_id,
        )

        if job.token_indices.numel() == 0:
            continue

        if expert_id == 7:
            out = boundary
        else:
            out = expert(
                job.expert_input
            )

        flat = merge_expert_result(
            flat,
            job,
            out,
        )

    h = (
        PRE_MOE_H
        + flat.reshape_as(MOE_INPUT)
    )

    for block in SERVER_MODEL.blocks[1:]:
        h, _, _ = block(h)

    h = SERVER_MODEL.final_norm(h)

    logits = SERVER_MODEL.lm_head(h)

    loss = F.cross_entropy(
        logits.reshape(-1, 64),
        Y.reshape(-1),
    )

    loss.backward()

    grad = boundary.grad.detach()

    save_file(
        {
            "gradient":
                grad.contiguous(),
        },
        str(GRADIENT),
    )

    GRADIENT_READY.set()

    return float(loss.detach())


def send_file(handler, path):
    data = path.read_bytes()

    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        "application/octet-stream",
    )
    handler.send_header(
        "Content-Length",
        str(len(data)),
    )
    handler.end_headers()

    handler.wfile.write(data)


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

    def do_GET(self):
        global PHYSICAL_START

        if self.path == "/job":
            if PHYSICAL_START is None:
                PHYSICAL_START = time.perf_counter()
            send_file(self, JOB)
            return

        if self.path == "/expert":
            send_file(
                self,
                Path(EXPERT_SHARD),
            )
            return

        if self.path == "/gradient":
            if not GRADIENT_READY.is_set():
                self.send_error(409)
                return

            send_file(
                self,
                GRADIENT,
            )
            return

        self.send_error(404)

    def do_POST(self):
        length = int(
            self.headers[
                "Content-Length"
            ]
        )

        payload = self.rfile.read(
            length
        )

        if self.path == "/forward":
            FORWARD.write_bytes(
                payload
            )
            FORWARD_EVENT.set()

        elif self.path == "/updated":
            global PHYSICAL_END

            UPDATED.write_bytes(
                payload
            )

            PHYSICAL_END = time.perf_counter()
            UPDATED_EVENT.set()

        else:
            self.send_error(404)
            return

        body = json.dumps({
            "status": "accepted",
            "bytes": length,
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
    ref_loss = prepare_split()

    print()
    print("FedEdgeMoE - M13H")
    print("Physical Split Backprop")
    print()
    print(
        "Reference loss:",
        f"{ref_loss:.8f}",
    )
    print(
        "Routed tokens:",
        TARGET_JOB.token_indices.numel(),
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

    if not FORWARD_EVENT.wait(300):
        raise RuntimeError(
            "No forward result"
        )

    split_loss = compute_gradient()

    if not UPDATED_EVENT.wait(300):
        raise RuntimeError(
            "No updated expert"
        )

    server.shutdown()

    physical = load_file(
        str(UPDATED)
    )

    max_diff = max(
        (
            REFERENCE_AFTER[key]
            - physical[key]
        ).abs().max().item()
        for key in REFERENCE_AFTER
    )

    print()
    print(
        "Split loss:",
        f"{split_loss:.8f}",
    )

    print(
        "Loss diff:",
        abs(
            ref_loss
            - split_loss
        ),
    )

    print(
        "Gradient payload:",
        GRADIENT.stat().st_size,
        "bytes",
    )

    print(
        "Updated expert payload:",
        UPDATED.stat().st_size,
        "bytes",
    )

    print(
        "Expert update max diff:",
        max_diff,
    )

    print()
    print(
        "PHYSICAL SPLIT TRAINING EQUIVALENT:",
        max_diff < 1e-5,
    )

    physical_seconds = (
        PHYSICAL_END - PHYSICAL_START
    )

    print(
        "PHYSICAL E2E:",
        f"{physical_seconds:.3f}s",
    )


if __name__ == "__main__":
    main()
