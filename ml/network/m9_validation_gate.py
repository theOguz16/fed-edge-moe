import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from safetensors.torch import load_file

import ml.network.m8_coordinator as base


START_VERSION = 30
MAX_ROUNDS = 5

# Fraction olarak 0.001 = 0.10 percentage point.
MIN_MEAN_IMPROVEMENT = 0.001

ROOT = Path("checkpoints/m9")
REPORT_PATH = Path(
    "results/m9_validation_gated_fl.json"
)


def reset_base_state():
    base.ROOT = ROOT

    base.STATE.update({
        "complete": False,
        "active": False,
        "round": None,
        "base_version": None,
        "new_version": None,
        "snapshot_tar": None,
        "snapshot_bytes": 0,
        "client_b_metadata": None,
        "client_b_delta_path": None,
    })

    base.CLIENT_B_EVENT.clear()


def save_report(history, final_version, early_stopped):
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "method":
                    "validation_gated_expert_fedavg",

                "start_version":
                    START_VERSION,

                "max_rounds":
                    MAX_ROUNDS,

                "min_mean_improvement":
                    MIN_MEAN_IMPROVEMENT,

                "final_version":
                    final_version,

                "early_stopped":
                    early_stopped,

                "history":
                    history,
            },
            f,
            indent=2,
        )


def main():
    reset_base_state()

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = base.get_device()

    server = ThreadingHTTPServer(
        (base.HOST, base.PORT),
        base.Handler,
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    server_thread.start()

    print()
    print("FedEdgeMoE - M9")
    print("Validation-Gated Physical FL")
    print()
    print(
        "Coordinator:",
        f"{base.HOST}:{base.PORT}",
    )
    print("Aggregator:", device)
    print(
        "Minimum accepted improvement:",
        f"{MIN_MEAN_IMPROVEMENT * 100:.2f} pp",
    )

    current_snapshot = (
        base.INITIAL_SNAPSHOT
    )

    current_version = (
        START_VERSION
    )

    current_model = (
        base.build_model()
    )

    base.load_global_snapshot(
        current_model,
        current_snapshot,
    )

    current_model = (
        current_model.to(device)
    )

    current_eval = (
        base.evaluate_all(
            current_model,
            device,
        )
    )

    current_mean = (
        base.mean_accuracy(
            current_eval
        )
    )

    base.print_eval(
        f"INITIAL GLOBAL V{current_version}",
        current_eval,
    )

    history = [{
        "round": 0,
        "version": current_version,
        "accepted": True,
        "evaluation": current_eval,
        "mean_accuracy": current_mean,
    }]

    early_stopped = False

    for round_number in range(
        1,
        MAX_ROUNDS + 1,
    ):
        round_start = (
            time.perf_counter()
        )

        candidate_version = (
            current_version + 1
        )

        print()
        print("=" * 78)
        print(
            f"ROUND {round_number}/{MAX_ROUNDS}"
        )
        print(
            f"BASE V{current_version} "
            f"-> CANDIDATE V{candidate_version}"
        )
        print("=" * 78)

        snapshot_tar = (
            base.create_snapshot_tar(
                current_snapshot,
                round_number,
            )
        )

        base.CLIENT_B_EVENT.clear()

        with base.LOCK:
            base.STATE["complete"] = False
            base.STATE["active"] = True

            base.STATE["round"] = (
                round_number
            )

            base.STATE["base_version"] = (
                current_version
            )

            base.STATE["new_version"] = (
                candidate_version
            )

            base.STATE["snapshot_tar"] = (
                snapshot_tar
            )

            base.STATE["snapshot_bytes"] = (
                snapshot_tar.stat().st_size
            )

            base.STATE[
                "client_b_metadata"
            ] = None

            base.STATE[
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
            base.train_client_a(
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
            "Waiting for physical MSI..."
        )

        received = (
            base.CLIENT_B_EVENT.wait(
                timeout=1200
            )
        )

        if not received:
            raise TimeoutError(
                "Physical client update timeout."
            )

        with base.LOCK:
            meta_b = dict(
                base.STATE[
                    "client_b_metadata"
                ]
            )

            delta_b_path = Path(
                base.STATE[
                    "client_b_delta_path"
                ]
            )

            base.STATE["active"] = False

        delta_b = load_file(
            str(delta_b_path)
        )

        similarity = (
            base.cosine_similarity(
                delta_a,
                delta_b,
            )
        )

        aggregated = (
            base.weighted_fedavg(
                delta_a,
                int(
                    meta_a[
                        "local_tokens_seen"
                    ]
                ),
                delta_b,
                int(
                    meta_b[
                        "local_tokens_seen"
                    ]
                ),
            )
        )

        candidate_model = (
            base.build_model()
        )

        base.load_global_snapshot(
            candidate_model,
            current_snapshot,
        )

        candidate_model = (
            candidate_model.to(device)
        )

        base.apply_expert_delta(
            candidate_model,
            aggregated,
        )

        candidate_eval = (
            base.evaluate_all(
                candidate_model,
                device,
            )
        )

        candidate_mean = (
            base.mean_accuracy(
                candidate_eval
            )
        )

        improvement = (
            candidate_mean
            - current_mean
        )

        accepted = (
            improvement
            >= MIN_MEAN_IMPROVEMENT
        )

        print()
        print(
            "Delta cosine similarity:",
            f"{similarity:+.6f}",
        )

        base.print_eval(
            (
                f"CANDIDATE "
                f"V{candidate_version}"
            ),
            candidate_eval,
        )

        print()
        print(
            "Mean improvement:",
            f"{improvement * 100:+.2f} pp",
        )

        print(
            "Validation decision:",
            "ACCEPT"
            if accepted
            else "REJECT",
        )

        if accepted:
            next_snapshot = (
                ROOT
                / f"global_v{candidate_version:04d}"
            )

            base.export_global_snapshot(
                model=candidate_model,
                output_dir=next_snapshot,
                global_version=
                    candidate_version,
            )

            current_snapshot = (
                next_snapshot
            )

            current_version = (
                candidate_version
            )

            current_eval = (
                candidate_eval
            )

            current_mean = (
                candidate_mean
            )

        else:
            rejected_path = (
                ROOT
                / (
                    "rejected_"
                    f"candidate_v"
                    f"{candidate_version:04d}"
                )
            )

            base.export_global_snapshot(
                model=candidate_model,
                output_dir=rejected_path,
                global_version=
                    candidate_version,
            )

            print()
            print(
                "Candidate rejected."
            )

            print(
                "Keeping:",
                f"Global V{current_version}",
            )

            print(
                "Rejected checkpoint:",
                rejected_path,
            )

            early_stopped = True

        history.append({
            "round":
                round_number,

            "base_version":
                meta_a[
                    "base_global_version"
                ],

            "candidate_version":
                candidate_version,

            "accepted":
                accepted,

            "delta_cosine_similarity":
                similarity,

            "previous_mean_accuracy":
                current_mean
                if not accepted
                else (
                    candidate_mean
                    - improvement
                ),

            "candidate_mean_accuracy":
                candidate_mean,

            "improvement":
                improvement,

            "evaluation":
                candidate_eval,

            "client_A":
                meta_a,

            "client_B":
                meta_b,

            "round_seconds":
                time.perf_counter()
                - round_start,
        })

        save_report(
            history,
            current_version,
            early_stopped,
        )

        if not accepted:
            break

    with base.LOCK:
        base.STATE["active"] = False
        base.STATE["complete"] = True

    print()
    print("=" * 78)
    print("M9 FINAL DECISION")
    print("=" * 78)

    print(
        "Best accepted version:",
        f"V{current_version}",
    )

    base.print_eval(
        "FINAL ACCEPTED GLOBAL",
        current_eval,
    )

    print()
    print(
        "Early stopped:",
        early_stopped,
    )

    print(
        "Final checkpoint:",
        current_snapshot,
    )

    print(
        "Report:",
        REPORT_PATH,
    )

    print()
    print(
        "Waiting briefly so worker can "
        "observe COMPLETE state..."
    )

    time.sleep(5)

    server.shutdown()

    print(
        "M9 VALIDATION-GATED FL COMPLETE"
    )


if __name__ == "__main__":
    main()
