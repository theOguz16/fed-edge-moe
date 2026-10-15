import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SOURCE = Path("results/benchmark_v2_plans")
OUT = SOURCE / "campaign31"
OUT.mkdir(parents=True, exist_ok=True)

MACHINES = {
    "mac_m4": {
        "accelerator": "mps",
        "threads": [1, 2, 4, 6, 8, 10],
    },
    "msi_3050": {
        "accelerator": "cuda",
        "threads": [1, 2, 4, 6, 8, 10, 12, 16],
    },
}

PILOT_PROTOCOL = {
    "size": 224,
    "batch": 1,
    "precision": "fp32",
    "weights": "random_none",
    "warmup": 4,
    "steps": 20,
    "repeats": 2,
}

FORMAL_PROTOCOL = {
    **PILOT_PROTOCOL,
    "warmup": 10,
    "steps": 100,
    "repeats": 3,
}

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows: {path}")
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def stable_hash(obj):
    raw = json.dumps(
        obj, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

core = read_csv(SOURCE / "vision_core30.csv")
extended = read_csv(SOURCE / "vision_extended30.csv")

core_ids = {r["candidate_id"] for r in core}
extended_ids = {r["candidate_id"] for r in extended}

if not core_ids.issubset(extended_ids):
    raise RuntimeError("Core is not a subset of Extended")

additions = [
    r for r in extended
    if r["candidate_id"] not in core_ids
]

# Select six representative placement classes per profile.
by_profile = defaultdict(list)
for row in core:
    by_profile[(row["model"], row["profile"])].append(row)

def centrality(row):
    cuts = [
        int(x[1:])
        for x in row["cut_after_groups"].split(",")
        if x
    ]
    center = (
        sum(cuts) / len(cuts)
        if cuts else 0
    )
    return abs(
        center - int(row["group_count"]) / 2
    )

pilot = []

for key, rows in sorted(by_profile.items()):
    selection = {
        "full_cpu": lambda r:
            r["category"] == "full_cpu",
        "full_gpu": lambda r:
            r["category"] == "full_gpu",
        "cpu_to_gpu": lambda r:
            r["category"] == "single_cut"
            and r["mapping"].startswith("C"),
        "gpu_to_cpu": lambda r:
            r["category"] == "single_cut"
            and r["mapping"].startswith("M"),
        "cpu_island": lambda r:
            r["category"] == "cpu_island",
        "gpu_island": lambda r:
            r["category"] == "gpu_island",
    }

    for label, predicate in selection.items():
        matches = [r for r in rows if predicate(r)]
        if not matches:
            raise RuntimeError(
                f"Missing pilot class {key} {label}"
            )

        chosen = min(matches, key=centrality)
        pilot.append({
            **chosen,
            "pilot_class": label
        })

assert len(pilot) == 36

def create_jobs(candidates, machine, tier):
    hardware = MACHINES[machine]

    is_pilot = tier == "pilot"
    threads = (
        [1, 4]
        if is_pilot else hardware["threads"]
    )
    process_runs = 1 if is_pilot else 3
    protocol = (
        PILOT_PROTOCOL
        if is_pilot else FORMAL_PROTOCOL
    )

    result = []

    for candidate in candidates:
        for t in threads:
            base = {
                "machine": machine,
                "model": candidate["model"],
                "profile": candidate["profile"],
                "manifest_sha256":
                    candidate["manifest_sha256"],
                "candidate_id":
                    candidate["candidate_id"],
                "mapping": candidate["mapping"],
                "accelerator": hardware["accelerator"],
                "threads": t,
                **protocol,
            }

            config_id = stable_hash(base)

            for process_run in range(process_runs):
                run_id = stable_hash({
                    "config_id": config_id,
                    "process_run": process_run,
                })

                result.append({
                    "run_id": run_id,
                    "config_id": config_id,
                    "tier": tier,
                    "machine": machine,
                    "model": candidate["model"],
                    "profile": candidate["profile"],
                    "candidate_id":
                        candidate["candidate_id"],
                    "mapping": candidate["mapping"],
                    "category": candidate["category"],
                    "pilot_class":
                        candidate.get("pilot_class", ""),
                    "cut_after_groups":
                        candidate["cut_after_groups"],
                    "accelerator":
                        hardware["accelerator"],
                    "threads": t,
                    "process_run": process_run,
                    "size": protocol["size"],
                    "batch": protocol["batch"],
                    "precision": protocol["precision"],
                    "weights": protocol["weights"],
                    "warmup": protocol["warmup"],
                    "steps": protocol["steps"],
                    "repeats": protocol["repeats"],
                    "manifest_sha256":
                        candidate["manifest_sha256"],
                    "status": "planned",
                })

    random.Random(20261015).shuffle(result)
    assert len({r["run_id"] for r in result}) == len(result)
    return result

print("=== MAC-31 CAMPAIGN PLAN ===")
print("Core candidates:", len(core))
print("Extended candidates:", len(extended))
print("Extended-only additions:", len(additions))
print("Pilot placements:", len(pilot))

for machine in MACHINES:
    print(f"\n{machine}")

    for tier, candidates in [
        ("pilot", pilot),
        ("core", core),
        ("extended_additions", additions),
    ]:
        jobs = create_jobs(candidates, machine, tier)
        dest = OUT / f"{machine}_{tier}.csv"
        write_csv(dest, jobs)

        print(
            f"  {tier:20s} "
            f"{len(jobs):6d} planned processes"
        )

print("\nAll plans are dry-run only.")
print("Saved:", OUT)
