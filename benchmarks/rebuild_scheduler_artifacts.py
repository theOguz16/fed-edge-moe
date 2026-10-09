import csv
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "benchmarks/build_scheduler_registry.py",
    "benchmarks/enrich_scheduler_memory.py",
    "benchmarks/summarize_scheduler_operating_ranges.py",
    "benchmarks/build_service_candidate_matching.py",
    "benchmarks/build_scheduler_quality_evidence.py",
    "benchmarks/match_scheduler_quality_evidence.py",
    "benchmarks/evaluate_service_feasibility.py",
    "benchmarks/test_quality_gate_contracts.py",
    "benchmarks/select_service_energy.py",
    "benchmarks/test_service_energy_selection.py",
]

for script in SCRIPTS:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, script], check=True)

base = Path("results/scheduler_candidate_registry.csv")
enriched = Path(
    "results/scheduler_candidate_registry_memory_enriched.csv"
)

def load(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

base_rows = load(base)
rows = load(enriched)

base_core = sum(
    is_true(r["scheduler_core"]) for r in base_rows
)

core = [r for r in rows if is_true(r["scheduler_core"])]

qwen3_core = [
    r for r in core
    if r["model"].lower() == "qwen3-1.7b"
]

assert len(base_rows) == len(rows), "Registry row count changed"
assert base_core == len(core), "Core candidate count changed"
assert qwen3_core, "No Qwen3 core candidates found"
assert all(
    is_true(r["has_any_memory_observation"])
    for r in qwen3_core
), "Qwen3 memory coverage incomplete"

memory_count = sum(
    is_true(r["has_any_memory_observation"])
    for r in core
)

print("\n=== REBUILD VALIDATION: PASS ===")
print(f"Configs          : {len(rows)}")
print(f"Core candidates  : {len(core)}")
print(f"Core with memory : {memory_count}")
print(f"Qwen3 memory     : {len(qwen3_core)}/{len(qwen3_core)}")
