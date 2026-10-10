import ast
import math
from pathlib import Path

SOURCE = Path("benchmarks/evaluate_service_feasibility.py")

NAMES = {
    "measured",
    "threshold",
    "throughput_gate",
    "latency_gate",
    "memory_contract",
    "memory_gate",
    "quality_gate",
    "overall_status",
}

# Load the actual function definitions without executing
# the script's top-level CSV reading/writing code.
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in NAMES
]

assert {node.name for node in functions} == NAMES

environment = {
    "math": math,
    "memory_contracts": {
        "vision_medium|Intel i7-11800H|cpu": {
            "unit": "MiB",
            "metric_field": "windows_peak_working_set_mib_max",
            "budget_mib": 1400,
        }
    },
    "quality_evidence": [
        {
            "service_id": "vision_easy",
            "device": "Apple M4",
            "backend": "mps",
            "precision": "fp16",
            "quantization": "",
            "dataset": "synthetic_imagenetv2",
            "metric": "top5_accuracy_pct",
            "direction": "higher",
            "evaluation_input_resolution": "224",
            "preprocessing": "synthetic_preprocessing",
            "value": "91.5",
        },
        {
            "service_id": "text_easy",
            "device": "Apple M4",
            "backend": "mps",
            "precision": "fp16",
            "quantization": "",
            "dataset": "synthetic_wikitext2",
            "metric": "perplexity",
            "direction": "lower",
            "evaluation_seq_len": "64",
            "evaluation_tokens": "1024",
            "value": "21.3",
        },
    ],
}

module = ast.Module(body=functions, type_ignores=[])
exec(compile(module, str(SOURCE), "exec"), environment)

throughput_gate = environment["throughput_gate"]
latency_gate = environment["latency_gate"]
memory_gate = environment["memory_gate"]
quality_gate = environment["quality_gate"]
overall_status = environment["overall_status"]

tests = []

def check(name, actual, expected):
    assert actual == expected, (
        f"{name}: expected={expected}, actual={actual}"
    )
    tests.append(name)
    print(f"{name:40s} PASS")

def expect_error(name, function, error_type):
    try:
        function()
    except error_type:
        tests.append(name)
        print(f"{name:40s} PASS")
    else:
        raise AssertionError(
            f"{name}: expected {error_type.__name__}"
        )

def updated(row, **changes):
    return {**row, **changes}

vision = {
    "service_id": "vision_easy",
    "device": "Apple M4",
    "backend": "mps",
    "precision": "fp16",
    "quantization": "",
    "modality": "vision",
    "resolution": "224",
    "throughput": "135",
    "throughput_unit": "images/s",
    "throughput_semantics": "images_per_second",
    "latency_sec": "0.025",
    "latency_semantics": "batch_completion",
    "quality_evidence_status": "SUPPORTED",
}

throughput_request = {
    "min_throughput": 120,
    "throughput_unit": "images/s",
    "throughput_semantics": "images_per_second",
}

latency_request = {
    "max_latency_s": 0.03,
    "latency_semantics": "batch_completion",
}

memory_row = {
    "service_id": "vision_medium",
    "device": "Intel i7-11800H",
    "backend": "cpu",
    "windows_peak_working_set_mib_max": "1315.7",
}

quality_request = {
    "min_quality": 91,
    "quality_metric": "top5_accuracy_pct",
    "quality_scope": "benchmark_reference",
    "quality_dataset": "synthetic_imagenetv2",
    "quality_preprocessing": "synthetic_preprocessing",
}

print("\n=== THROUGHPUT TESTS ===")

check(
    "throughput sufficient",
    throughput_gate(vision, throughput_request),
    "PASS",
)
check(
    "throughput insufficient",
    throughput_gate(
        vision,
        updated(throughput_request, min_throughput=150),
    ),
    "FAIL",
)
check(
    "throughput unit mismatch",
    throughput_gate(
        vision,
        updated(throughput_request, throughput_unit="tokens/s"),
    ),
    "UNKNOWN",
)
check(
    "throughput missing observation",
    throughput_gate(
        updated(vision, throughput=""),
        throughput_request,
    ),
    "UNKNOWN",
)
check(
    "throughput not configured",
    throughput_gate(vision, {}),
    "NOT_CONFIGURED",
)

print("\n=== LATENCY TESTS ===")

check(
    "latency within deadline",
    latency_gate(vision, latency_request),
    "PASS",
)
check(
    "latency exceeds deadline",
    latency_gate(
        vision,
        updated(latency_request, max_latency_s=0.02),
    ),
    "FAIL",
)
check(
    "latency semantics mismatch",
    latency_gate(
        vision,
        updated(
            latency_request,
            latency_semantics="per_request",
        ),
    ),
    "UNKNOWN",
)
check(
    "latency missing observation",
    latency_gate(
        updated(vision, latency_sec=""),
        latency_request,
    ),
    "UNKNOWN",
)
check(
    "latency not configured",
    latency_gate(vision, {}),
    "NOT_CONFIGURED",
)

print("\n=== MEMORY TESTS ===")

check(
    "memory within budget",
    memory_gate(memory_row),
    "PASS",
)

contracts = environment["memory_contracts"]
contract = contracts["vision_medium|Intel i7-11800H|cpu"]

contract["budget_mib"] = 1300

check(
    "memory exceeds budget",
    memory_gate(memory_row),
    "FAIL",
)

contract["budget_mib"] = 1400

check(
    "memory observation missing",
    memory_gate(
        updated(
            memory_row,
            windows_peak_working_set_mib_max="",
        )
    ),
    "UNKNOWN",
)

contract["budget_mib"] = None

check(
    "memory not configured",
    memory_gate(memory_row),
    "NOT_CONFIGURED",
)

contract["budget_mib"] = 1400
contract["unit"] = "MB"

expect_error(
    "memory unit mismatch rejected",
    lambda: memory_gate(memory_row),
    ValueError,
)

contract["unit"] = "MiB"

print("\n=== VISION QUALITY TESTS ===")

check(
    "quality top5 sufficient",
    quality_gate(vision, quality_request),
    "PASS",
)
check(
    "quality top5 insufficient",
    quality_gate(
        vision,
        updated(quality_request, min_quality=92),
    ),
    "FAIL",
)
check(
    "quality conditional evidence",
    quality_gate(
        updated(
            vision,
            quality_evidence_status="CONDITIONAL",
        ),
        quality_request,
    ),
    "UNKNOWN",
)
check(
    "quality resolution mismatch",
    quality_gate(
        updated(vision, resolution="320"),
        quality_request,
    ),
    "UNKNOWN",
)
check(
    "quality preprocessing mismatch",
    quality_gate(
        vision,
        updated(
            quality_request,
            quality_preprocessing="different_preprocessing",
        ),
    ),
    "UNKNOWN",
)
check(
    "quality dataset mismatch",
    quality_gate(
        vision,
        updated(
            quality_request,
            quality_dataset="different_dataset",
        ),
    ),
    "UNKNOWN",
)
check(
    "quality scope missing",
    quality_gate(
        vision,
        updated(quality_request, quality_scope="deployment"),
    ),
    "UNKNOWN",
)
check(
    "quality precision mismatch",
    quality_gate(
        updated(vision, precision="fp32"),
        quality_request,
    ),
    "UNKNOWN",
)
check(
    "quality not configured",
    quality_gate(vision, {}),
    "NOT_CONFIGURED",
)

expect_error(
    "quality conflicting thresholds rejected",
    lambda: quality_gate(
        vision,
        updated(quality_request, max_quality=99),
    ),
    ValueError,
)

print("\n=== TEXT QUALITY TESTS ===")

text_row = {
    "service_id": "text_easy",
    "device": "Apple M4",
    "backend": "mps",
    "precision": "fp16",
    "quantization": "",
    "modality": "text",
    "quality_evidence_status": "SUPPORTED",
}

text_request = {
    "max_quality": 22,
    "quality_metric": "perplexity",
    "quality_scope": "benchmark_reference",
    "quality_dataset": "synthetic_wikitext2",
    "quality_eval_seq_len": 64,
    "quality_eval_tokens": 1024,
}

check(
    "perplexity within maximum",
    quality_gate(text_row, text_request),
    "PASS",
)
check(
    "perplexity exceeds maximum",
    quality_gate(
        text_row,
        updated(text_request, max_quality=20),
    ),
    "FAIL",
)
check(
    "perplexity evaluation protocol mismatch",
    quality_gate(
        text_row,
        updated(text_request, quality_eval_seq_len=128),
    ),
    "UNKNOWN",
)

print("\n=== OVERALL FEASIBILITY TESTS ===")

all_pass = {
    "throughput": "PASS",
    "latency": "PASS",
    "memory": "PASS",
    "quality": "PASS",
}

check(
    "all gates pass",
    overall_status(all_pass),
    "FEASIBLE",
)
check(
    "one known failure",
    overall_status(updated(all_pass, quality="FAIL")),
    "INFEASIBLE",
)
check(
    "one unknown gate",
    overall_status(updated(all_pass, quality="UNKNOWN")),
    "UNVERIFIED",
)
check(
    "one unconfigured gate",
    overall_status(
        updated(all_pass, memory="NOT_CONFIGURED")
    ),
    "UNVERIFIED",
)
check(
    "FAIL overrides UNKNOWN",
    overall_status(
        updated(
            all_pass,
            latency="FAIL",
            quality="UNKNOWN",
        )
    ),
    "INFEASIBLE",
)

print()
print("=== STEP 8 FEASIBILITY REGRESSION ===")
print("Tests passed      :", len(tests))
print("Actual gate code  : YES")
print("Synthetic fixtures: YES")
print("Original CSV files: UNMODIFIED")
print("STEP 8 FEASIBILITY GATES: PASS")
