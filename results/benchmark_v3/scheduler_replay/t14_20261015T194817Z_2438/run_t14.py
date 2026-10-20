import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()
BASE = Path("results/benchmark_v3")

T11 = BASE / (
    "thread_partition_convnext/t11_20261015T191957Z_2438"
)
T12 = BASE / (
    "holdout_thread_confirmation/t12_20261015T192923Z_2438"
)
T13 = BASE / (
    "research_synthesis/t13_20261015T194336Z_2438"
)
T05 = BASE / (
    "validation/t05_20261015T173344Z_2438"
)

THREADS = (1, 4)
POLICIES = (
    "FULL_MPS",
    "ALWAYS_HYBRID",
    "T11_SELECTED",
    "HINDSIGHT_ORACLE",
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def readcsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def writecsv(path, rows):
    assert rows
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        w.writeheader()
        w.writerows(rows)

def savejson(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

def verify(directory):
    path = directory / "checksums.json"
    checks = json.loads(path.read_text())
    assert checks

    for filename, expected in checks.items():
        target = directory / filename
        assert target.is_file(), str(target)
        assert sha(target) == expected, str(target)

def verify_campaign(directory, worker_count):
    verify(directory)

    for n in range(1, worker_count + 1):
        verify(directory / f"process_{n:02d}")

def close(a, b):
    return math.isclose(
        float(a), float(b),
        rel_tol=1e-10,
        abs_tol=1e-8,
    )

print("=== SOURCE VERIFICATION ===", flush=True)

verify_campaign(T11, 18)
verify_campaign(T12, 8)
verify(T13)
verify(T05)

index = readcsv(T13 / "source_inventory.csv")

for campaign, path in [
    ("T11_convnext_threads", T11 / "paired_comparison.csv"),
    ("T12_holdout_confirmation", T12 / "paired_images.csv"),
]:
    selected = [
        row for row in index
        if row["campaign"] == campaign
        and row["relative_path"] == str(path)
    ]
    assert len(selected) == 1
    assert selected[0]["sha256"] == sha(path)

print("T11/T12 SOURCE CHECKSUMS: PASS")
print("T13 SOURCE INDEX CROSS-CHECK: PASS")

# --------------------------------------------------
# Training: T11 only.
# The policy is frozen before reading test comparisons.
# --------------------------------------------------

train = readcsv(T11 / "paired_comparison.csv")
assert len(train) == 72

decisions = {}
training_details = {}

for threads in THREADS:
    records = [
        row for row in train
        if row["model"] == "convnext_large"
        and int(row["threads"]) == threads
        and row["mapping"] == "MPS_TO_CPU"
    ]

    assert len(records) == 3
    assert {
        int(r["round"]) for r in records
    } == {1, 2, 3}

    deltas = []

    for row in records:
        baseline = float(row["full_mps_ms"])
        hybrid = float(row["hybrid_ms"])
        delta = 100 * (hybrid / baseline - 1)

        assert close(delta, row["delta_percent"])
        assert int(row["hybrid_faster"]) == int(
            hybrid < baseline
        )
        deltas.append(delta)

    training_median = statistics.median(deltas)

    # Fixed, deterministic rule, using T11 only.
    decision = (
        "MPS_TO_CPU"
        if training_median < 0
        else "FULL_MPS"
    )

    decisions[threads] = decision
    training_details[str(threads)] = {
        "thread_count": threads,
        "independent_training_processes": 3,
        "training_paired_deltas_percent": deltas,
        "training_median_delta_percent": training_median,
        "selected_mapping": decision,
        "selection_rule": (
            "MPS_TO_CPU if median paired percent "
            "difference < 0; otherwise FULL_MPS"
        ),
    }

assert decisions == {
    1: "MPS_TO_CPU",
    4: "FULL_MPS",
}

policy_path = OUT / "frozen_policy.json"
savejson(policy_path, {
    "protocol": "T14_retrospective_offline_replay_v1",
    "model": "convnext_large",
    "training_campaign": str(T11),
    "evaluation_campaign": str(T12),
    "training_only_source_for_selection": True,
    "selection_rule": (
        "Per thread count, select the lower-median "
        "T11 paired-percent-delta placement"
    ),
    "policy_by_thread": {
        str(k): v for k, v in decisions.items()
    },
    "training_details": training_details,
    "caveat": (
        "Retrospective analysis: T12 outcomes were "
        "previously inspected. Not a fresh blind test."
    ),
})

print("\n=== FROZEN T11 POLICY ===")
for threads in THREADS:
    item = training_details[str(threads)]
    print(
        f"Threads={threads}: "
        f"selected={decisions[threads]}, "
        f"training median="
        f"{item['training_median_delta_percent']:+.3f}%"
    )

# --------------------------------------------------
# Evaluation: fixed decisions on T12 comparisons.
# --------------------------------------------------

test = readcsv(T12 / "paired_images.csv")
holdout = json.loads(
    (T12 / "holdout_images.json").read_text()
)
validation = json.loads(
    (T05 / "metadata.json").read_text()
)

assert len(test) == 128
assert len(holdout) == 16

prior_paths = {
    row["path"] for row in validation["images"]
}
prior_hashes = {
    row["sha256"] for row in validation["images"]
}

assert not (
    {r["path"] for r in holdout} & prior_paths
)
assert not (
    {r["sha256"] for r in holdout} & prior_hashes
)

seen = set()
raw_replay = []
by_process_policy = defaultdict(list)
by_policy_thread = defaultdict(list)
orders = defaultdict(list)

for row in test:
    threads = int(row["threads"])
    round_no = int(row["round"])
    image_index = int(row["image_index"])

    assert threads in THREADS
    assert round_no in (1, 2, 3, 4)
    assert 0 <= image_index < 16

    key = (threads, round_no, image_index)
    assert key not in seen
    seen.add(key)

    image_info = holdout[image_index]
    assert row["image_path"] == image_info["path"]
    assert row["image_sha256"] == image_info["sha256"]

    full = float(row["full_mps_ms"])
    hybrid = float(row["hybrid_ms"])

    assert full > 0 and hybrid > 0
    assert close(
        100 * (hybrid / full - 1),
        row["delta_percent"]
    )

    assert int(row["hybrid_faster"]) == int(
        hybrid < full
    )

    first = int(row["full_mps_order"])
    assert first in (1, 2)

    orders[(threads, image_index)].append(first)

    realized = {
        "FULL_MPS": full,
        "ALWAYS_HYBRID": hybrid,
        "T11_SELECTED": (
            hybrid
            if decisions[threads] == "MPS_TO_CPU"
            else full
        ),
        "HINDSIGHT_ORACLE": min(full, hybrid),
    }

    for policy in POLICIES:
        chosen = realized[policy]
        oracle = min(full, hybrid)

        replay_row = {
            "threads": threads,
            "round": round_no,
            "image_index": image_index,
            "image_sha256": row["image_sha256"],
            "policy": policy,
            "training_selected_mapping": decisions[threads],
            "full_mps_ms": full,
            "hybrid_ms": hybrid,
            "policy_ms": chosen,
            "oracle_ms": oracle,
            "delta_vs_full_mps_percent": (
                100 * (chosen / full - 1)
            ),
            "regret_vs_hindsight_oracle_ms": (
                chosen - oracle
            ),
            "faster_than_full_mps": int(chosen < full),
            "slower_than_full_mps": int(chosen > full),
        }

        assert replay_row[
            "regret_vs_hindsight_oracle_ms"
        ] >= -1e-9

        raw_replay.append(replay_row)
        by_process_policy[
            (threads, round_no, policy)
        ].append(replay_row)
        by_policy_thread[
            (threads, policy)
        ].append(replay_row)

assert len(seen) == 128
assert len(raw_replay) == 512

for t in THREADS:
    for idx in range(16):
        assert sorted(orders[(t, idx)]) == [
            1, 1, 2, 2
        ]

print("\nHOLDOUT DISJOINTNESS: PASS")
print("BALANCED PLACEMENT ORDER: PASS")
print("PAIRED TEST COMPARISONS: 128/128 PASS")

# --------------------------------------------------
# Per-process results: each round is a distinct
# Python process for the given thread setting.
# --------------------------------------------------

process_rows = []

for threads in THREADS:
    for round_no in range(1, 5):
        for policy in POLICIES:
            rows = by_process_policy[
                (threads, round_no, policy)
            ]
            assert len(rows) == 16

            deltas = [
                r["delta_vs_full_mps_percent"]
                for r in rows
            ]

            baseline_sum = sum(
                r["full_mps_ms"] for r in rows
            )
            policy_sum = sum(
                r["policy_ms"] for r in rows
            )

            process_rows.append({
                "threads": threads,
                "round": round_no,
                "policy": policy,
                "image_pairs": 16,
                "median_paired_delta_percent":
                    statistics.median(deltas),
                "summed_latency_delta_percent":
                    100 * (policy_sum / baseline_sum - 1),
                "faster_pairs": sum(
                    r["faster_than_full_mps"]
                    for r in rows
                ),
                "slower_pairs": sum(
                    r["slower_than_full_mps"]
                    for r in rows
                ),
                "summed_oracle_regret_ms": sum(
                    r["regret_vs_hindsight_oracle_ms"]
                    for r in rows
                ),
            })

assert len(process_rows) == 32

summary_rows = []

for threads in THREADS:
    for policy in POLICIES:
        rows = by_policy_thread[(threads, policy)]
        assert len(rows) == 64

        process = [
            r for r in process_rows
            if r["threads"] == threads
            and r["policy"] == policy
        ]
        assert len(process) == 4

        base_sum = sum(
            r["full_mps_ms"] for r in rows
        )
        policy_sum = sum(
            r["policy_ms"] for r in rows
        )

        summary_rows.append({
            "threads": threads,
            "policy": policy,
            "image_process_pairs": len(rows),
            "independent_processes": 4,
            "faster_pairs": sum(
                r["faster_than_full_mps"]
                for r in rows
            ),
            "slower_pairs": sum(
                r["slower_than_full_mps"]
                for r in rows
            ),
            "median_of_process_median_deltas_percent":
                statistics.median(
                    r["median_paired_delta_percent"]
                    for r in process
                ),
            "faster_processes": sum(
                r["median_paired_delta_percent"] < 0
                for r in process
            ),
            "summed_latency_delta_percent":
                100 * (policy_sum / base_sum - 1),
            "summed_hindsight_regret_ms": sum(
                r["regret_vs_hindsight_oracle_ms"]
                for r in rows
            ),
        })

assert len(summary_rows) == 8

print("\n=== T14 OFFLINE POLICY REPLAY ===")
for threads in THREADS:
    print(f"\nCPU THREADS={threads}")
    print(
        f"{'Policy':<19} "
        f"{'Round median Δ':>15} "
        f"{'Wins':>9}"
    )

    for policy in POLICIES:
        row = next(
            r for r in summary_rows
            if r["threads"] == threads
            and r["policy"] == policy
        )
        print(
            f"{policy:<19} "
            f"{row['median_of_process_median_deltas_percent']:>+14.3f}% "
            f"{row['faster_processes']:>7}/4"
        )

report = [
    "# T14 — Retrospective Offline Scheduler Replay",
    "",
    "## Study status",
    "",
    "This is a retrospective analysis of existing "
    "measurements, not a new blind experiment and "
    "not an online scheduler benchmark.",
    "",
    "The placement policy was computed exclusively "
    "from T11 independent-process paired latency "
    "results, then applied unchanged to T12 pairs.",
    "",
    "## Training decisions",
    "",
    "| CPU threads | T11 median delta | Selected placement |",
    "|---:|---:|---|",
]

for threads in THREADS:
    item = training_details[str(threads)]
    report.append(
        f"| {threads} | "
        f"{item['training_median_delta_percent']:+.3f}% | "
        f"{decisions[threads]} |"
    )

report.extend([
    "",
    "Negative percentages favor the hybrid.",
    "",
    "## Existing T12 holdout replay",
    "",
    "| Threads | Policy | Median of four process deltas | "
    "Faster processes | Faster image-process pairs |",
    "|---:|---|---:|---:|---:|",
])

for row in summary_rows:
    report.append(
        f"| {row['threads']} | {row['policy']} | "
        f"{row['median_of_process_median_deltas_percent']:+.3f}% | "
        f"{row['faster_processes']}/4 | "
        f"{row['faster_pairs']}/64 |"
    )

report.extend([
    "",
    "## Interpretation",
    "",
    "The T11-selected policy chooses cut 5 for one "
    "CPU thread and Full MPS for four CPU threads.",
    "",
    "For the one-thread setting, the T12 holdout "
    "process medians do not confirm the training "
    "advantage. This demonstrates a deployment "
    "risk of choosing placement solely from a "
    "small exploratory latency sample.",
    "",
    "The hindsight oracle uses latency values that "
    "would not be available to an online scheduler. "
    "It is an optimistic descriptive bound, not a "
    "deployable algorithm or evidence of achievable "
    "real-time speedup.",
    "",
    "Reported summed latency is merely the sum of "
    "measured batch-1 calls across the fixed image "
    "set. It does not measure actual throughput, "
    "switching cost, scheduling overhead or overlap.",
    "",
    "The same sixteen images recur in four "
    "independent processes per thread count. "
    "Inference calls and image-process pairs must "
    "not be treated as wholly independent samples.",
    "",
    "The run has no measurements of energy, thermal "
    "state, peak allocated memory, heterogeneous "
    "edge-node communication or concurrent workload.",
    "",
    "## Research decision",
    "",
    "Before claiming resource-aware scheduling gains, "
    "a prospective protocol should freeze the "
    "decision rule, collect fresh test workloads, "
    "and evaluate latency alongside measured "
    "resource constraints and scheduler overhead.",
])

policy_report = OUT / "offline_replay_report.md"
replay_path = OUT / "replay_per_pair.csv"
process_path = OUT / "replay_per_process.csv"
summary_path = OUT / "replay_summary.csv"
metadata_path = OUT / "metadata.json"

writecsv(replay_path, raw_replay)
writecsv(process_path, process_rows)
writecsv(summary_path, summary_rows)

policy_report.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

savejson(metadata_path, {
    "protocol": "T14_retrospective_offline_replay_v1",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "model": "convnext_large",
    "training_source": str(T11),
    "training_pair_sha256": sha(
        T11 / "paired_comparison.csv"
    ),
    "evaluation_source": str(T12),
    "evaluation_pair_sha256": sha(
        T12 / "paired_images.csv"
    ),
    "synthesis_source": str(T13),
    "script_sha256": sha(__file__),
    "fixed_thread_counts": list(THREADS),
    "evaluation_images": 16,
    "evaluation_independent_processes": 8,
    "evaluated_pairs": 128,
    "policy_comparison_rows": 512,
    "process_policy_rows": 32,
    "summary_rows": 8,
    "blind_prospective_evaluation": False,
    "actual_scheduler_executed": False,
    "no_new_inference_performed": True,
    "resource_aware_objective_evaluated": False,
})

artifacts = [
    Path(__file__),
    policy_path,
    replay_path,
    process_path,
    summary_path,
    policy_report,
    metadata_path,
]

savejson(
    OUT / "checksums.json",
    {p.name: sha(p) for p in artifacts},
)

verify(OUT)

assert len(readcsv(replay_path)) == 512
assert len(readcsv(process_path)) == 32
assert len(readcsv(summary_path)) == 8

print("\nT14 REPLAY INTEGRITY: PASS")
print("TRAINING POLICY: FROZEN FROM T11")
print("EVALUATION PAIRS: 128")
print("POLICY COMPARISON ROWS: 512")
print("NEW INFERENCE CALLS: 0")
print("RESULT DIRECTORY:", OUT)
