import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REGISTRY = Path(
    "results/scheduler_candidate_registry_memory_enriched.csv"
)
REQUIREMENTS = Path("configs/service_requirements.json")
OUT = Path("results/service_candidate_matching.csv")

services = json.loads(
    REQUIREMENTS.read_text(encoding="utf-8")
)["services"]

with REGISTRY.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    registry_columns = reader.fieldnames
    rows = list(reader)

def model_id(value):
    name = value.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]", "", name)

def number(value):
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))

def is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")

def shape_matches(row, shape, task):
    if task == "vision_classification":
        return (
            number(row["resolution"]) == shape["resolution"]
            and number(row["batch"]) == shape["batch"]
        )

    return (
        number(row["context_tokens"]) == shape["context"]
        and number(row["output_tokens"]) == shape["output"]
        and number(row["batch"]) == shape["batch"]
    )

LATENCY_CONTRACTS = {
    "vision_easy": "batch_inference_completion",
    "vision_medium": "batch_inference_completion",
    "vision_hard": "batch_inference_completion",
    "text_easy": "batch_generate_call_completion",
    "text_medium": "batch_generate_call_completion",
    "text_hard": "llamacpp_batched_benchmark_total",
}

contracts = {
    "vision_classification": "post_warmup_batch_aggregate",
    "text_pytorch": "post_warmup_generate_call_batch_aggregate",
    "text_llamacpp": "generation_phase_s_tg",
}

matches = []
coverage = defaultdict(list)
matched_registry_indices = set()

for service_id, service in services.items():
    task = service["task_type"]

    for profile, request in service["profiles"].items():
        shape = request["input_shape"]

        for index, row in enumerate(rows):
            if not is_true(row["scheduler_core"]):
                continue

            if model_id(row["model"]) != model_id(service["model"]):
                continue

            if not shape_matches(row, shape, task):
                continue

            if task == "vision_classification":
                contract = contracts["vision_classification"]
            elif row["backend"].startswith("llama.cpp") or (
                row["model"] == "Qwen3-1.7B"
            ):
                contract = contracts["text_llamacpp"]
            else:
                contract = contracts["text_pytorch"]

            output_row = {
                "service_id": service_id,
                "request_profile": profile,
                "throughput_semantics": contract,
                "latency_semantics": (
                    LATENCY_CONTRACTS[service_id]
                    if (row.get("latency_sec") or "").strip()
                    else ""
                ),
                **row,
            }

            matches.append(output_row)
            coverage[(service_id, profile)].append(output_row)

            if index in matched_registry_indices:
                raise RuntimeError(
                    f"Registry candidate matched multiple requests: {index}"
                )
            matched_registry_indices.add(index)

for service_id, service in services.items():
    for profile in service["profiles"]:
        if not coverage[(service_id, profile)]:
            raise RuntimeError(
                f"No candidates: {service_id}/{profile}"
            )

assert sum(
    bool(r["latency_semantics"]) for r in matches
) == 54

assert all(
    bool(r["latency_semantics"]) == bool(r["latency_sec"])
    for r in matches
)

for r in matches:
    if not r["latency_semantics"]:
        continue

    sources = r["source_files"]

    if r["service_id"] == "text_hard":
        assert r["backend"] == "llama.cpp/metal"
        assert (
            "qwen3_hard_q4_mac_accel_power_synced.csv"
            in sources
        )
    else:
        assert (
            "sweep.csv" in sources
            or "scaling.csv" in sources
        )

fieldnames = [
    "service_id",
    "request_profile",
    "throughput_semantics",
    "latency_semantics",
    *registry_columns,
]

with OUT.open(newline="", mode="w", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=fieldnames, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(matches)

latency_count = sum(bool(r["latency_sec"]) for r in matches)
memory_count = sum(
    is_true(r["has_any_memory_observation"]) for r in matches
)
joint_count = sum(
    bool(r["latency_sec"])
    and is_true(r["has_any_memory_observation"])
    for r in matches
)

assert len(coverage) == 18
assert len(matches) == 57
assert latency_count == 54
assert memory_count == 54
assert joint_count == 51

print("=== SERVICE-CANDIDATE MATCHING: PASS ===")
print("Service profiles      :", len(coverage))
print("Matched candidates    :", len(matches))
print("With latency          :", latency_count)
print("With memory           :", memory_count)
print("With both             :", joint_count)

print("\nSERVICE PROFILE COVERAGE")
print(f"{'SERVICE':17s} {'PROFILE':8s} {'N':>3s} {'LAT':>4s} {'MEM':>4s}")

for (service_id, profile), group in sorted(coverage.items()):
    lat = sum(bool(r["latency_sec"]) for r in group)
    mem = sum(
        is_true(r["has_any_memory_observation"])
        for r in group
    )

    print(
        f"{service_id:17s} {profile:8s} "
        f"{len(group):3d} {lat:4d} {mem:4d}"
    )

print(f"\nSaved: {OUT}")
