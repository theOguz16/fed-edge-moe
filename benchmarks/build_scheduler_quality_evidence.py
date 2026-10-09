import csv
from pathlib import Path

ROOT = Path("results")
OUT = ROOT / "scheduler_quality_evidence.csv"

fields = [
    "service_id",
    "model",
    "device",
    "backend",
    "precision",
    "quantization",
    "task",
    "dataset",
    "evaluation_batch",
    "evaluation_input_resolution",
    "preprocessing",
    "evaluation_seq_len",
    "evaluation_tokens",
    "metric",
    "value",
    "direction",
    "applicability",
    "source_file",
]

evidence = []

def read_one(filename):
    path = ROOT / filename
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one summary row: {filename}"
        )
    return rows[0]

def add(**kwargs):
    record = {key: "" for key in fields}
    record.update(kwargs)
    evidence.append(record)

VISION = {
    "resnet50": "vision_easy",
    "convnext_base": "vision_medium",
    "convnext_large": "vision_hard",
}

VISION_WEIGHTS = {
    "resnet50": "ResNet50_Weights.IMAGENET1K_V2",
    "convnext_base": "ConvNeXt_Base_Weights.IMAGENET1K_V1",
    "convnext_large": "ConvNeXt_Large_Weights.IMAGENET1K_V1",
}

for model, service in VISION.items():
    for platform, device, backend in (
        ("mac", "Apple M4", "mps"),
        ("msi", "RTX 3050 Laptop", "cuda"),
    ):
        filename = (
            f"{model}_imagenetv2_quality_"
            f"{platform}_summary.csv"
        )
        row = read_one(filename)

        assert row["model"] == model
        assert int(row["images"]) == 10000

        for precision in ("fp32", "fp16"):
            for metric, suffix in (
                ("top1_accuracy_pct", "top1_accuracy_pct"),
                ("top5_accuracy_pct", "top5_accuracy_pct"),
            ):
                add(
                    service_id=service,
                    model=model,
                    device=device,
                    backend=backend,
                    precision=precision,
                    quantization="none",
                    task="vision_classification",
                    dataset=row["dataset"],
                    evaluation_batch=row["batch_size"],
                    evaluation_input_resolution=224,
                    preprocessing=(
                        VISION_WEIGHTS[model]
                        + "|resize=232|center_crop=224"
                        + "|bilinear|imagenet_mean_std"
                    ),
                    metric=metric,
                    value=row[f"{precision}_{suffix}"],
                    direction="higher",
                    applicability="evaluation_protocol_only",
                    source_file=filename,
                )

TEXT = [
    (
        "distilgpt2_text_precision_quality_mac.csv",
        "text_easy", "Apple M4", "mps",
    ),
    (
        "distilgpt2_text_precision_quality_msi.csv",
        "text_easy", "RTX 3050 Laptop", "cuda",
    ),
    (
        "qwen25_text_precision_quality_mac.csv",
        "text_medium", "Apple M4", "mps",
    ),
    (
        "qwen3_text_precision_quality_mac.csv",
        "text_hard", "Apple M4", "mps",
    ),
]

for filename, service, device, backend in TEXT:
    row = read_one(filename)

    for precision in ("fp32", "fp16"):
        add(
            service_id=service,
            model=row["model_id"],
            device=device,
            backend=backend,
            precision=precision,
            quantization="none",
            task="text_language_model_evaluation",
            dataset="Salesforce/wikitext:wikitext-2-raw-v1:test",
            evaluation_seq_len=row["seq_len"],
            evaluation_tokens=row["eval_tokens"],
            metric="perplexity",
            value=row[f"{precision}_perplexity"],
            direction="lower",
            applicability="native_precision_evaluation_only",
            source_file=filename,
        )

SELFCHECK = [
    (
        "qwen25_selfcheck_mac_summary.csv",
        "text_medium",
    ),
    (
        "qwen3_selfcheck_mac_summary.csv",
        "text_hard",
    ),
]

for filename, service in SELFCHECK:
    row = read_one(filename)

    for precision in ("fp32", "fp16"):
        add(
            service_id=service,
            model=row["model"],
            device="Apple M4",
            backend="mps",
            precision=precision,
            quantization="none",
            task="factual_consistency_evaluation",
            dataset=(
                f"prompt_set_{row['prompts']}_prompts_"
                f"{row['samples_per_prompt']}_samples"
            ),
            metric="mean_selfcheck_nli",
            value=row[f"{precision}_mean_selfcheck_nli"],
            direction="lower",
            applicability="native_precision_evaluation_only",
            source_file=filename,
        )

assert len(evidence) == 36, len(evidence)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(evidence)

print("QUALITY EVIDENCE REGISTRY: PASS")
print("Vision metric records :", 24)
print("Text PPL records      :", 8)
print("SelfCheck records     :", 4)
print("Total records         :", len(evidence))
print("Saved                 :", OUT)
print()
print("NOTE: No scheduler quality feasibility decisions assigned.")
