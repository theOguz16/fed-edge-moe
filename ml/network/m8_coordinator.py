import json
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
    export_global_snapshot,
)


HOST = "0.0.0.0"
PORT = 8765

START_VERSION = 30
NUM_ROUNDS = 5

TARGET_LAYER = 0
TARGET_EXPERT = 6

LOCAL_STEPS = 120
BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 5e-4

INITIAL_SNAPSHOT = Path(
    "checkpoints/m5_candidates/global_step_0030"
)

ROOT = Path("checkpoints/m8")
RESULT_PATH = Path(
    "results/m8_multiround_physical.json"
)

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
    "D2 (+3)",
    "D3 (+5)",
]

DOMAIN_STEPS = [1, 2, 3, 5]

LOCK = threading.Lock()
CLIENT_B_EVENT = threading.Event()

STATE = {
    "complete": False,
    "active": False,
    "round": None,
    "base_version": None,
    "new_version": None,
    "snapshot_tar": None,
    "snapshot_bytes": 0,
    "client_b_metadata": None,
    "client_b_delta_path": None,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    device,
):
    starts = torch.randint(
        0,
        16,
        (BATCH_SIZE,),
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


def build_eval_set(
    domain,
    device,
):
    starts = torch.arange(
        0,
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

        prediction = output.logits.argmax(
            dim=-1
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


def mean_accuracy(results):
    return (
        sum(
            x["accuracy"]
            for x in results
        )
        / len(results)
    )


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


def flatten_delta(delta):
    return torch.cat([
        delta[key].float().reshape(-1)
        for key in sorted(delta)
    ])


def cosine_similarity(a, b):
    va = flatten_delta(a)
    vb = flatten_delta(b)

    return float(
        torch.nn.functional.cosine_similarity(
            va.unsqueeze(0),
            vb.unsqueeze(0),
        ).item()
    )


def train_client_a(
    snapshot,
    round_number,
    base_version,
    device,
):
    torch.manual_seed(
        8000 + round_number
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
        .blocks[TARGET_LAYER]
        .moe
        .experts[TARGET_EXPERT]
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

    before = clone_expert(
        expert
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    start = time.perf_counter()

    model.train()

    for step in range(
        1,
        LOCAL_STEPS + 1,
    ):
        x, y = generate_domain_batch(
            0,
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

    training_seconds = (
        time.perf_counter() - start
    )

    after_eval = evaluate_all(
        model,
        device,
    )

    after = clone_expert(
        expert
    )

    delta = calculate_delta(
        before,
        after,
    )

    round_dir = (
        ROOT
        / f"round_{round_number:02d}"
        / "client_A"
    )

    round_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    delta_path = (
        round_dir
        / "expert_delta.safetensors"
    )

    save_file(
        delta,
        str(delta_path),
    )

    metadata = {
        "client_id":
            "client_A_mac_mps",

        "round":
            round_number,

        "domain":
            0,

        "base_global_version":
            base_version,

        "layer":
            TARGET_LAYER,

        "expert":
            TARGET_EXPERT,

        "local_steps":
            LOCAL_STEPS,

        "batch_size":
            BATCH_SIZE,

        "local_tokens_seen":
            LOCAL_STEPS
            * BATCH_SIZE
            * SEQ_LEN,

        "trainable_parameters":
            sum(
                p.numel()
                for p in trainable
            ),

        "training_seconds":
            training_seconds,

        "delta_l2_norm":
            delta_norm(delta),

        "delta_bytes":
            delta_path.stat().st_size,

        "before":
            before_eval,

        "after":
            after_eval,
    }

    return (
        delta,
        metadata,
    )


def create_snapshot_tar(
    snapshot,
    round_number,
):
    round_dir = (
        ROOT
        / f"round_{round_number:02d}"
    )

    round_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tar_path = (
        round_dir
        / "global_snapshot.tar.gz"
    )

    with tarfile.open(
        tar_path,
        "w:gz",
    ) as tar:
        tar.add(
            snapshot,
            arcname="global_snapshot",
        )

    return tar_path


def weighted_fedavg(
    delta_a,
    weight_a,
    delta_b,
    weight_b,
):
    total = (
        weight_a + weight_b
    )

    return {
        key: (
            (
                delta_a[key] * weight_a
                + delta_b[key] * weight_b
            )
            / total
        ).contiguous()
        for key in delta_a
    }


def apply_expert_delta(
    model,
    delta,
):
    expert = (
        model
        .blocks[TARGET_LAYER]
        .moe
        .experts[TARGET_EXPERT]
    )

    state = expert.state_dict()

    updated = {
        key: (
            state[key]
            + delta[key].to(
                device=state[key].device,
                dtype=state[key].dtype,
            )
        )
        for key in state
    }

    expert.load_state_dict(
        updated
    )


def send_json(
    handler,
    payload,
    status=200,
):
    data = json.dumps(
        payload,
        indent=2,
    ).encode("utf-8")

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "application/json",
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
            self.client_address[0],
            fmt % args,
        )

    def do_GET(self):
        if self.path == "/health":
            with LOCK:
                response = {
                    "status": "ok",
                    "complete":
                        STATE["complete"],
                    "active":
                        STATE["active"],
                    "round":
                        STATE["round"],
                    "base_version":
                        STATE["base_version"],
                }

            send_json(
                self,
                response,
            )
            return

        if self.path == "/job":
            with LOCK:
                if STATE["complete"]:
                    response = {
                        "complete": True,
                    }

                elif not STATE["active"]:
                    response = {
                        "complete": False,
                        "ready": False,
                    }

                else:
                    response = {
                        "complete": False,
                        "ready": True,

                        "round":
                            STATE["round"],

                        "base_global_version":
                            STATE["base_version"],

                        "new_global_version":
                            STATE["new_version"],

                        "target_layer":
                            TARGET_LAYER,

                        "target_expert":
                            TARGET_EXPERT,

                        "local_domain":
                            1,

                        "local_steps":
                            LOCAL_STEPS,

                        "batch_size":
                            BATCH_SIZE,

                        "snapshot_bytes":
                            STATE["snapshot_bytes"],
                    }

            send_json(
                self,
                response,
            )
            return

        if self.path == "/snapshot":
            with LOCK:
                tar_path = (
                    STATE["snapshot_tar"]
                )

            if (
                tar_path is None
                or not Path(tar_path).exists()
            ):
                send_json(
                    self,
                    {"error": "snapshot unavailable"},
                    status=404,
                )
                return

            data = Path(
                tar_path
            ).read_bytes()

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "application/gzip",
            )

            self.send_header(
                "Content-Length",
                str(len(data)),
            )

            self.end_headers()

            self.wfile.write(data)
            return

        if self.path == "/worker.py":
            worker_path = Path(
                "ml/network/m8_worker_loop.py"
            )

            if not worker_path.exists():
                send_json(
                    self,
                    {"error": "worker file missing"},
                    status=404,
                )
                return

            data = worker_path.read_bytes()

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/x-python",
            )

            self.send_header(
                "Content-Length",
                str(len(data)),
            )

            self.end_headers()

            self.wfile.write(data)
            return

        send_json(
            self,
            {"error": "not found"},
            status=404,
        )

    def do_POST(self):
        length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        body = self.rfile.read(
            length
        )

        if self.path == "/upload/client_B/metadata":
            try:
                metadata = json.loads(
                    body.decode("utf-8")
                )
            except Exception:
                send_json(
                    self,
                    {"error": "invalid json"},
                    status=400,
                )
                return

            with LOCK:
                expected_round = (
                    STATE["round"]
                )

                expected_version = (
                    STATE["base_version"]
                )

                active = STATE["active"]

            if not active:
                send_json(
                    self,
                    {"error": "no active round"},
                    status=409,
                )
                return

            if (
                metadata.get("round")
                != expected_round
                or
                metadata.get(
                    "base_global_version"
                )
                != expected_version
            ):
                send_json(
                    self,
                    {
                        "error":
                            "stale or wrong-version update",

                        "expected_round":
                            expected_round,

                        "expected_base_version":
                            expected_version,
                    },
                    status=409,
                )
                return

            if (
                metadata.get("layer")
                != TARGET_LAYER
                or
                metadata.get("expert")
                != TARGET_EXPERT
            ):
                send_json(
                    self,
                    {
                        "error":
                            "wrong expert assignment"
                    },
                    status=409,
                )
                return

            round_dir = (
                ROOT
                / f"round_{expected_round:02d}"
                / "client_B"
            )

            round_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata_path = (
                round_dir
                / "metadata.json"
            )

            metadata_path.write_bytes(
                body
            )

            with LOCK:
                STATE[
                    "client_b_metadata"
                ] = metadata

            send_json(
                self,
                {
                    "status": "metadata accepted",
                    "round": expected_round,
                    "base_version":
                        expected_version,
                },
            )
            return

        if self.path == "/upload/client_B/delta":
            with LOCK:
                round_number = (
                    STATE["round"]
                )

                metadata = (
                    STATE[
                        "client_b_metadata"
                    ]
                )

                active = (
                    STATE["active"]
                )

            if (
                not active
                or metadata is None
            ):
                send_json(
                    self,
                    {
                        "error":
                            "metadata must be accepted first"
                    },
                    status=409,
                )
                return

            round_dir = (
                ROOT
                / f"round_{round_number:02d}"
                / "client_B"
            )

            round_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            delta_path = (
                round_dir
                / "expert_delta.safetensors"
            )

            delta_path.write_bytes(
                body
            )

            with LOCK:
                STATE[
                    "client_b_delta_path"
                ] = delta_path

            CLIENT_B_EVENT.set()

            send_json(
                self,
                {
                    "status": "delta accepted",
                    "round": round_number,
                    "bytes": len(body),
                },
            )
            return

        send_json(
            self,
            {"error": "not found"},
            status=404,
        )


def print_eval(
    title,
    results,
):
    print()
    print(title)

    for index, result in enumerate(
        results
    ):
        print(
            f"{DOMAIN_NAMES[index]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc={result['accuracy'] * 100:6.2f}%"
        )

    print(
        f"MEAN     "
        f"acc={mean_accuracy(results) * 100:6.2f}%"
    )


def main():
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    print()
    print("FedEdgeMoE - M8")
    print("Versioned Multi-Round Physical FL")
    print()
    print("Coordinator:", f"{HOST}:{PORT}")
    print("Aggregator:", device)
    print(
        f"Rounds: {NUM_ROUNDS}"
    )
    print()

    current_snapshot = (
        INITIAL_SNAPSHOT
    )

    current_version = (
        START_VERSION
    )

    history = []

    initial_model = build_model()

    load_global_snapshot(
        initial_model,
        current_snapshot,
    )

    initial_model = (
        initial_model.to(device)
    )

    initial_eval = evaluate_all(
        initial_model,
        device,
    )

    print_eval(
        f"INITIAL GLOBAL V{current_version}",
        initial_eval,
    )

    history.append({
        "version":
            current_version,

        "round":
            0,

        "evaluation":
            initial_eval,

        "mean_accuracy":
            mean_accuracy(
                initial_eval
            ),
    })

    for round_number in range(
        1,
        NUM_ROUNDS + 1,
    ):
        round_start = (
            time.perf_counter()
        )

        new_version = (
            current_version + 1
        )

        print()
        print("=" * 78)
        print(
            f"ROUND {round_number}/{NUM_ROUNDS}"
        )
        print(
            f"GLOBAL V{current_version} "
            f"-> V{new_version}"
        )
        print("=" * 78)

        snapshot_tar = (
            create_snapshot_tar(
                current_snapshot,
                round_number,
            )
        )

        CLIENT_B_EVENT.clear()

        with LOCK:
            STATE["complete"] = False
            STATE["active"] = True

            STATE["round"] = (
                round_number
            )

            STATE["base_version"] = (
                current_version
            )

            STATE["new_version"] = (
                new_version
            )

            STATE["snapshot_tar"] = (
                snapshot_tar
            )

            STATE["snapshot_bytes"] = (
                snapshot_tar.stat().st_size
            )

            STATE[
                "client_b_metadata"
            ] = None

            STATE[
                "client_b_delta_path"
            ] = None

        print(
            "Snapshot bytes:",
            f"{snapshot_tar.stat().st_size:,}",
        )

        print()
        print(
            "Training Client A "
            "(Mac / D0)..."
        )

        delta_a, meta_a = (
            train_client_a(
                current_snapshot,
                round_number,
                current_version,
                device,
            )
        )

        print(
            "Client A complete:",
            f"{meta_a['training_seconds']:.3f}s",
        )

        print(
            "Waiting for physical "
            "MSI Client B..."
        )

        received = (
            CLIENT_B_EVENT.wait(
                timeout=1200
            )
        )

        if not received:
            raise TimeoutError(
                "MSI Client B did not "
                "submit an update."
            )

        with LOCK:
            meta_b = dict(
                STATE[
                    "client_b_metadata"
                ]
            )

            delta_b_path = Path(
                STATE[
                    "client_b_delta_path"
                ]
            )

            STATE["active"] = False

        delta_b = load_file(
            str(delta_b_path)
        )

        weight_a = int(
            meta_a[
                "local_tokens_seen"
            ]
        )

        weight_b = int(
            meta_b[
                "local_tokens_seen"
            ]
        )

        similarity = (
            cosine_similarity(
                delta_a,
                delta_b,
            )
        )

        aggregated = (
            weighted_fedavg(
                delta_a,
                weight_a,
                delta_b,
                weight_b,
            )
        )

        aggregate_start = (
            time.perf_counter()
        )

        global_model = build_model()

        load_global_snapshot(
            global_model,
            current_snapshot,
        )

        global_model = (
            global_model.to(device)
        )

        apply_expert_delta(
            global_model,
            aggregated,
        )

        new_eval = evaluate_all(
            global_model,
            device,
        )

        next_snapshot = (
            ROOT
            / f"global_v{new_version:04d}"
        )

        export_global_snapshot(
            model=global_model,
            output_dir=next_snapshot,
            global_version=new_version,
        )

        aggregation_seconds = (
            time.perf_counter()
            - aggregate_start
        )

        round_seconds = (
            time.perf_counter()
            - round_start
        )

        print()
        print(
            "Delta cosine similarity:",
            f"{similarity:+.6f}",
        )

        print_eval(
            f"GLOBAL V{new_version}",
            new_eval,
        )

        print()
        print(
            "Client A training:",
            f"{meta_a['training_seconds']:.3f}s",
        )

        print(
            "Client B training:",
            f"{meta_b['training_seconds']:.3f}s",
        )

        print(
            "Client B snapshot download:",
            f"{meta_b['snapshot_download_bytes']:,} bytes",
            f"in {meta_b['snapshot_download_seconds']:.3f}s",
        )

        print(
            "Client B delta upload:",
            f"{meta_b['delta_bytes']:,} bytes",
        )

        print(
            "Aggregation + checkpoint:",
            f"{aggregation_seconds:.3f}s",
        )

        print(
            "Round total:",
            f"{round_seconds:.3f}s",
        )

        history.append({
            "round":
                round_number,

            "base_version":
                current_version,

            "version":
                new_version,

            "evaluation":
                new_eval,

            "mean_accuracy":
                mean_accuracy(
                    new_eval
                ),

            "delta_cosine_similarity":
                similarity,

            "client_A":
                meta_a,

            "client_B":
                meta_b,

            "aggregation_seconds":
                aggregation_seconds,

            "round_seconds":
                round_seconds,

            "snapshot_bytes":
                snapshot_tar.stat().st_size,
        })

        current_snapshot = (
            next_snapshot
        )

        current_version = (
            new_version
        )

        with open(
            RESULT_PATH,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "start_version":
                        START_VERSION,

                    "num_rounds":
                        NUM_ROUNDS,

                    "history":
                        history,
                },
                f,
                indent=2,
            )

    with LOCK:
        STATE["active"] = False
        STATE["complete"] = True

    print()
    print("=" * 78)
    print("M8 MULTI-ROUND CONVERGENCE")
    print("=" * 78)

    print(
        f"{'Version':<12}"
        f"{'D0':>9}"
        f"{'D1':>9}"
        f"{'D2':>9}"
        f"{'D3':>9}"
        f"{'Mean':>9}"
    )

    for entry in history:
        results = entry["evaluation"]

        accuracies = [
            r["accuracy"] * 100
            for r in results
        ]

        print(
            f"V{entry['version']:<11}"
            f"{accuracies[0]:>8.2f}%"
            f"{accuracies[1]:>8.2f}%"
            f"{accuracies[2]:>8.2f}%"
            f"{accuracies[3]:>8.2f}%"
            f"{sum(accuracies)/4:>8.2f}%"
        )

    print()
    print(
        "Final checkpoint:",
        current_snapshot,
    )

    print(
        "Report:",
        RESULT_PATH,
    )

    print()
    print(
        "VERSIONED MULTI-ROUND "
        "PHYSICAL FL COMPLETE"
    )

    server.shutdown()


if __name__ == "__main__":
    main()
