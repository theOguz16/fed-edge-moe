import csv
from collections import Counter
from pathlib import Path

ROOT = Path("results")

CANDIDATES = ROOT / "service_candidate_matching.csv"
EVIDENCE = ROOT / "scheduler_quality_evidence.csv"
OUT = ROOT / "service_candidate_quality_applicability.csv"

def load(path):
    with path.open(
        newline="", encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

candidate_fields, candidates = load(CANDIDATES)
_, evidence = load(EVIDENCE)

extra_fields = [
    "quality_evidence_status",
    "quality_reference_scope",
    "quality_evidence_metrics",
    "quality_evidence_sources",
    "quality_evaluation_datasets",
    "quality_application_gate",
]

output = []

for candidate in candidates:
    related = [
        e for e in evidence
        if (
            e["service_id"] == candidate["service_id"]
            and e["precision"] == candidate["precision"]
            and e["quantization"] == candidate["quantization"]
        )
    ]

    exact = [
        e for e in related
        if (
            e["device"] == candidate["device"]
            and e["backend"] == candidate["backend"]
        )
    ]

    if exact:
        status = "SUPPORTED"
        reference = exact
    elif related:
        status = "CONDITIONAL"
        reference = related
    else:
        status = "UNVERIFIED"
        reference = []

    if candidate["modality"] == "vision":
        resolution = int(float(candidate["resolution"]))
        batch = int(float(candidate["batch"]))

        if status == "UNVERIFIED":
            scope = "no_matching_precision_quality_evidence"
        elif status == "CONDITIONAL":
            scope = (
                "cross_device_or_backend_reference;"
                "evaluated_resolution=224;"
                f"request_resolution={resolution};"
                f"request_batch={batch}"
            )
        else:
            eval_batch = reference[0]["evaluation_batch"]
            scope = (
                "same_device_backend_precision;"
                "evaluated_resolution=224;"
                f"request_resolution={resolution};"
                f"evaluation_batch={eval_batch};"
                f"request_batch={batch};"
                "deployment_preprocessing_not_verified"
            )

    elif candidate["precision"] == "q4_k_m":
        if status == "SUPPORTED":
            scope = (
                "native_q4_k_m_same_device_backend;"
                "llamacpp_quality_evaluation;"
                "not_comparable_to_hf_seq64_perplexity;"
                "service_generation_quality_not_verified"
            )
        elif status == "CONDITIONAL":
            scope = (
                "cross_device_or_backend_q4_k_m_reference;"
                "quality_transfer_not_verified"
            )
        else:
            scope = "no_matching_q4_k_m_quality_evidence"

    elif candidate["precision"] == "default_uncontrolled":
        scope = (
            "scheduler_runtime_precision_not_verified;"
            "cannot_assign_fp32_or_fp16_evidence"
        )

    elif status == "CONDITIONAL":
        scope = (
            "cross_device_or_backend_text_quality_reference;"
            "evaluation_not_equivalent_to_service_generation"
        )

    elif status == "SUPPORTED":
        scope = (
            "same_device_backend_precision;"
            "wikitext2_test_seq64_and_or_factual_prompt_evaluation;"
            "service_generation_quality_not_verified"
        )

    else:
        scope = "no_matching_quality_evidence"

    record = dict(candidate)

    record.update({
        "quality_evidence_status": status,
        "quality_reference_scope": scope,
        "quality_evidence_metrics": ";".join(
            sorted({e["metric"] for e in reference})
        ),
        "quality_evidence_sources": ";".join(
            sorted({e["source_file"] for e in reference})
        ),
        "quality_evaluation_datasets": ";".join(
            sorted({e["dataset"] for e in reference})
        ),
        "quality_application_gate": "NOT_ASSESSED",
    })

    output.append(record)

assert len(output) == 57

counts = Counter(
    r["quality_evidence_status"] for r in output
)

assert sum(counts.values()) == 57

with OUT.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=candidate_fields + extra_fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output)

print("=== QUALITY APPLICABILITY: PASS ===")
print("Service candidates :", len(output))
print("Direct evidence    :", counts["SUPPORTED"])
print("Conditional        :", counts["CONDITIONAL"])
print("Unverified         :", counts["UNVERIFIED"])
print("Quality gates PASS :", 0)
print("Saved              :", OUT)

print("\nBY SERVICE")
for service in sorted({
    r["service_id"] for r in output
}):
    rows = [
        r for r in output
        if r["service_id"] == service
    ]
    statuses = Counter(
        r["quality_evidence_status"] for r in rows
    )
    print(
        f"{service:15s} "
        f"N={len(rows):2d} "
        f"supported={statuses['SUPPORTED']:2d} "
        f"conditional={statuses['CONDITIONAL']:2d} "
        f"unverified={statuses['UNVERIFIED']:2d}"
    )
