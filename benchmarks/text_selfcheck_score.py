import argparse
import csv
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    sentences = [
        p.strip()
        for p in parts
        if len(p.strip()) >= 8
    ]

    if not sentences:
        sentences = [text]

    return sentences


def score_file(path, checker):
    data = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    scored_records = []
    all_sentence_scores = []

    records = data["records"]

    for i, record in enumerate(records):
        print(
            f"{Path(path).name}: "
            f"{i + 1}/{len(records)}"
        )

        sentences = split_sentences(
            record["main_answer"]
        )

        if not sentences:
            raise RuntimeError(
                f"No sentences for prompt {i}"
            )

        scores = checker.predict(
            sentences=sentences,
            sampled_passages=record["samples"],
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        all_sentence_scores.extend(
            scores.tolist()
        )

        scored_records.append({
            "prompt_idx":
                record["prompt_idx"],
            "prompt":
                record["prompt"],
            "main_answer":
                record["main_answer"],
            "num_sentences":
                len(sentences),
            "mean_selfcheck_nli":
                float(scores.mean()),
            "max_selfcheck_nli":
                float(scores.max()),
            "sentence_scores":
                scores.tolist(),
            "samples":
                record["samples"],
        })

    all_scores = np.asarray(
        all_sentence_scores,
        dtype=np.float64,
    )

    prompt_scores = np.asarray(
        [
            r["mean_selfcheck_nli"]
            for r in scored_records
        ],
        dtype=np.float64,
    )

    summary = {
        "model":
            data["model"],
        "precision":
            data["precision"],
        "device":
            data["device"],
        "prompts":
            data["prompts"],
        "samples_per_prompt":
            data["samples_per_prompt"],
        "sentences":
            len(all_scores),
        "mean_selfcheck_nli":
            float(all_scores.mean()),
        "median_selfcheck_nli":
            float(np.median(all_scores)),
        "max_selfcheck_nli":
            float(all_scores.max()),
        "mean_prompt_score":
            float(prompt_scores.mean()),
        "generation_time_s":
            data["generation_time_s"],
    }

    return summary, scored_records


def save_detail(path, records):
    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = [
            "prompt_idx",
            "prompt",
            "main_answer",
            "num_sentences",
            "mean_selfcheck_nli",
            "max_selfcheck_nli",
            "sentence_scores",
            "samples",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for r in records:
            row = dict(r)
            row["sentence_scores"] = repr(
                row["sentence_scores"]
            )
            row["samples"] = repr(
                row["samples"]
            )
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp32",
        required=True,
    )

    parser.add_argument(
        "--fp16",
        required=True,
    )

    parser.add_argument(
        "--tag",
        required=True,
    )

    args = parser.parse_args()

    print(
        "Loading SelfCheckGPT-NLI on CPU..."
    )

    checker = SelfCheckNLI(
        device=torch.device("cpu")
    )

    summary32, records32 = score_file(
        args.fp32,
        checker,
    )

    summary16, records16 = score_file(
        args.fp16,
        checker,
    )

    if (
        summary32["model"]
        != summary16["model"]
    ):
        raise RuntimeError(
            "FP32/FP16 model mismatch"
        )

    model = summary32["model"]

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    detail32 = out_dir / (
        f"{model}_selfcheck_"
        f"fp32_{args.tag}.csv"
    )

    detail16 = out_dir / (
        f"{model}_selfcheck_"
        f"fp16_{args.tag}.csv"
    )

    save_detail(
        detail32,
        records32,
    )

    save_detail(
        detail16,
        records16,
    )

    scores32 = np.asarray(
        [
            r["mean_selfcheck_nli"]
            for r in records32
        ],
        dtype=np.float64,
    )

    scores16 = np.asarray(
        [
            r["mean_selfcheck_nli"]
            for r in records16
        ],
        dtype=np.float64,
    )

    comparison = {
        "model": model,
        "prompts":
            summary32["prompts"],
        "samples_per_prompt":
            summary32[
                "samples_per_prompt"
            ],

        "fp32_mean_selfcheck_nli":
            summary32[
                "mean_selfcheck_nli"
            ],

        "fp16_mean_selfcheck_nli":
            summary16[
                "mean_selfcheck_nli"
            ],

        "delta_fp16_minus_fp32":
            float(
                summary16[
                    "mean_selfcheck_nli"
                ]
                - summary32[
                    "mean_selfcheck_nli"
                ]
            ),

        "mean_abs_prompt_score_drift":
            float(
                np.abs(
                    scores16
                    - scores32
                ).mean()
            ),

        "fp32_generation_time_s":
            summary32[
                "generation_time_s"
            ],

        "fp16_generation_time_s":
            summary16[
                "generation_time_s"
            ],
    }

    summary_path = out_dir / (
        f"{model}_selfcheck_"
        f"{args.tag}_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                comparison.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(comparison)

    print("\nSELFCHECKGPT SUMMARY")
    print("-" * 72)

    for k, v in comparison.items():
        print(
            f"{k:38s}: {v}"
        )

    print("\nSaved:", detail32)
    print("Saved:", detail16)
    print("Saved:", summary_path)

    del checker
    gc.collect()


if __name__ == "__main__":
    main()
