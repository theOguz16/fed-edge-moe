import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("results")
BASE = json.loads(
    Path("configs/service_requirements.json").read_text(
        encoding="utf-8"
    )
)

def load(filename):
    with (ROOT / filename).open(
        newline="", encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))

evidence = load("scheduler_quality_evidence.csv")

def find_evidence(service, device, precision, metric):
    matches = [
        row for row in evidence
        if row["service_id"] == service
        and row["device"] == device
        and row["precision"] == precision
        and row["metric"] == metric
    ]
    assert len(matches) == 1, (service, device, precision, metric)
    return matches[0]

vision = find_evidence(
    "vision_easy", "Apple M4", "fp16", "top1_accuracy_pct"
)
qwen25 = find_evidence(
    "text_medium", "Apple M4", "fp16", "perplexity"
)
qwen3 = find_evidence(
    "text_hard", "Apple M4", "fp16", "perplexity"
)
distil = find_evidence(
    "text_easy", "Apple M4", "fp16", "perplexity"
)

def vision_requirement(limit, preprocessing=None):
    return {
        "quality_metric": "top1_accuracy_pct",
        "min_quality": limit,
        "quality_scope": "benchmark_reference",
        "quality_dataset": vision["dataset"],
        "quality_preprocessing": (
            vision["preprocessing"]
            if preprocessing is None else preprocessing
        ),
    }

def text_requirement(reference, limit):
    return {
        "quality_metric": "perplexity",
        "max_quality": limit,
        "quality_scope": "benchmark_reference",
        "quality_dataset": reference["dataset"],
        "quality_eval_seq_len": int(
            reference["evaluation_seq_len"]
        ),
        "quality_eval_tokens": int(
            reference["evaluation_tokens"]
        ),
    }

v = float(vision["value"])
q = float(qwen25["value"])

cases = [
    (
        "VISION_224_PASS", "vision_easy", "medium",
        "Apple M4", "fp16",
        vision_requirement(v - 0.1), "PASS",
    ),
    (
        "VISION_224_FAIL", "vision_easy", "medium",
        "Apple M4", "fp16",
        vision_requirement(v + 0.1), "FAIL",
    ),
    (
        "VISION_160_UNKNOWN", "vision_easy", "light",
        "Apple M4", "fp16",
        vision_requirement(v - 0.1), "UNKNOWN",
    ),
    (
        "WRONG_PREPROCESSING_UNKNOWN", "vision_easy", "medium",
        "Apple M4", "fp16",
        vision_requirement(
            v - 0.1, vision["preprocessing"] + "|incorrect"
        ), "UNKNOWN",
    ),
    (
        "QWEN25_MAC_PPL_PASS", "text_medium", "medium",
        "Apple M4", "fp16",
        text_requirement(qwen25, q + 0.1), "PASS",
    ),
    (
        "QWEN25_MAC_PPL_FAIL", "text_medium", "medium",
        "Apple M4", "fp16",
        text_requirement(qwen25, q - 0.1), "FAIL",
    ),
    (
        "QWEN25_CUDA_CROSS_DEVICE_UNKNOWN",
        "text_medium", "medium", "RTX 3050 Laptop", "fp16",
        text_requirement(qwen25, q + 0.1), "UNKNOWN",
    ),
    (
        "QWEN3_Q4_UNKNOWN", "text_hard", "medium",
        "Apple M4", "q4_k_m",
        text_requirement(
            qwen3, float(qwen3["value"]) + 0.1
        ), "UNKNOWN",
    ),
    (
        "DISTIL_UNCONTROLLED_UNKNOWN", "text_easy", "medium",
        "Apple M4", "default_uncontrolled",
        text_requirement(
            distil, float(distil["value"]) + 0.1
        ), "UNKNOWN",
    ),
]

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)

    for index, case in enumerate(cases):
        (
            label, service, profile, device,
            precision, settings, expected
        ) = case

        config = copy.deepcopy(BASE)
        config["services"][service]["profiles"][profile].update(
            settings
        )

        requirements_file = directory / f"requirements_{index}.json"
        output_file = directory / f"result_{index}.csv"

        requirements_file.write_text(
            json.dumps(config), encoding="utf-8"
        )

        subprocess.run(
            [
                sys.executable,
                "benchmarks/evaluate_service_feasibility.py",
                "--requirements", str(requirements_file),
                "--output", str(output_file),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        with output_file.open(
            newline="", encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))

        selected = [
            row for row in rows
            if row["service_id"] == service
            and row["request_profile"] == profile
            and row["device"] == device
            and row["precision"] == precision
        ]

        assert len(selected) == 1, label

        actual = selected[0]["quality_gate"]
        assert actual == expected, (label, expected, actual)

        expected_overall = (
            "INFEASIBLE" if expected == "FAIL"
            else "UNVERIFIED"
        )

        assert (
            selected[0]["feasibility_status"]
            == expected_overall
        ), label

        print(f"{label}: {actual} — PASS")

print()
print("QUALITY GATE REGRESSION TESTS: PASS")
print("Tests completed:", len(cases))
print("Real service requirements changed: NO")
