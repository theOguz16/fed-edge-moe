import argparse
import csv
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = {
    "distilgpt2": "distilgpt2",
    "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3": "Qwen/Qwen3-1.7B",
}


def get_device(name):
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_tokens(tokenizer, eval_tokens):
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    text = "\n".join(
        row["text"]
        for row in ds
        if row["text"].strip()
    )

    required = eval_tokens + 1

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=required,
    ).input_ids[0]

    if len(ids) < required:
        raise RuntimeError(
            f"Need {required} tokens, found {len(ids)}"
        )

    return ids[:required]


def run_precision(
    model_name,
    model_id,
    precision,
    token_ids,
    device,
    seq_len,
    kl_sample_indices,
    reference_logp=None,
):
    print(
        f"\nLoading {model_name} "
        f"{precision} on {device}..."
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id
    )

    model.eval()
    model.to(device)

    if precision == "fp16":
        model.half()
    else:
        model.float()

    nll_values = []
    entropy_values = []
    predictions = []

    reference_samples = {}
    kl_values = []

    total_tokens = len(token_ids) - 1

    sync(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        for start in range(
            0,
            total_tokens,
            seq_len,
        ):
            end = min(
                start + seq_len,
                total_tokens,
            )

            seq = token_ids[
                start:end + 1
            ]

            inp = seq[:-1].unsqueeze(0).to(
                device
            )

            target = seq[1:].to(
                device
            )

            logits = model(
                input_ids=inp
            ).logits[0]

            # Compare distributions in FP32,
            # regardless of model execution dtype.
            logp = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            nll = -logp.gather(
                1,
                target.unsqueeze(1),
            ).squeeze(1)

            probs = logp.exp()

            entropy = -(
                probs * logp
            ).sum(dim=1)

            pred = logits.argmax(
                dim=-1
            )

            nll_values.extend(
                nll.cpu().tolist()
            )

            entropy_values.extend(
                entropy.cpu().tolist()
            )

            predictions.extend(
                pred.cpu().tolist()
            )

            for local_idx in range(
                end - start
            ):
                global_idx = (
                    start + local_idx
                )

                if (
                    global_idx
                    not in kl_sample_indices
                ):
                    continue

                current = (
                    logp[local_idx]
                    .detach()
                    .cpu()
                    .float()
                )

                if reference_logp is None:
                    reference_samples[
                        global_idx
                    ] = current
                else:
                    ref = reference_logp[
                        global_idx
                    ]

                    ref_prob = ref.exp()

                    kl = (
                        ref_prob
                        * (ref - current)
                    ).sum().item()

                    kl_values.append(kl)

            done = end

            if (
                done % 512 < seq_len
                or done == total_tokens
            ):
                print(
                    f"{precision}: "
                    f"{done}/{total_tokens} tokens"
                )

            del (
                logits,
                logp,
                probs,
                nll,
                entropy,
                pred,
            )

    sync(device)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return {
        "nll": np.asarray(
            nll_values,
            dtype=np.float64,
        ),
        "entropy": np.asarray(
            entropy_values,
            dtype=np.float64,
        ),
        "pred": np.asarray(
            predictions,
            dtype=np.int64,
        ),
        "reference_logp":
            reference_samples,
        "kl":
            np.asarray(
                kl_values,
                dtype=np.float64,
            ),
        "time_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        required=True,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--kl-samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--tag",
        required=True,
    )

    args = parser.parse_args()

    device = get_device(
        args.device
    )

    model_id = MODELS[
        args.model
    ]

    print("Model:", model_id)
    print("Device:", device)
    print(
        "Evaluation tokens:",
        args.eval_tokens,
    )
    print(
        "Sequence length:",
        args.seq_len,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_id
        )
    )

    token_ids = build_tokens(
        tokenizer,
        args.eval_tokens,
    )

    sample_count = min(
        args.kl_samples,
        args.eval_tokens,
    )

    sample_indices = set(
        np.linspace(
            0,
            args.eval_tokens - 1,
            sample_count,
            dtype=int,
        ).tolist()
    )

    fp32 = run_precision(
        args.model,
        model_id,
        "fp32",
        token_ids,
        device,
        args.seq_len,
        sample_indices,
    )

    fp16 = run_precision(
        args.model,
        model_id,
        "fp16",
        token_ids,
        device,
        args.seq_len,
        sample_indices,
        reference_logp=(
            fp32[
                "reference_logp"
            ]
        ),
    )

    ppl32 = math.exp(
        float(
            fp32["nll"].mean()
        )
    )

    ppl16 = math.exp(
        float(
            fp16["nll"].mean()
        )
    )

    agreement = (
        fp32["pred"]
        == fp16["pred"]
    ).mean() * 100.0

    entropy_drift = np.abs(
        fp32["entropy"]
        - fp16["entropy"]
    )

    nll_drift = np.abs(
        fp32["nll"]
        - fp16["nll"]
    )

    summary = {
        "model": args.model,
        "model_id": model_id,
        "device": str(device),
        "eval_tokens":
            args.eval_tokens,
        "seq_len":
            args.seq_len,
        "kl_samples":
            len(fp16["kl"]),

        "fp32_perplexity":
            ppl32,
        "fp16_perplexity":
            ppl16,
        "perplexity_delta":
            ppl16 - ppl32,

        "token_top1_agreement_pct":
            agreement,

        "mean_kl_fp32_to_fp16":
            float(
                fp16["kl"].mean()
            ),

        "max_kl_fp32_to_fp16":
            float(
                fp16["kl"].max()
            ),

        "mean_entropy_fp32":
            float(
                fp32["entropy"].mean()
            ),

        "mean_entropy_fp16":
            float(
                fp16["entropy"].mean()
            ),

        "mean_entropy_drift":
            float(
                entropy_drift.mean()
            ),

        "max_entropy_drift":
            float(
                entropy_drift.max()
            ),

        "mean_nll_drift":
            float(
                nll_drift.mean()
            ),

        "max_nll_drift":
            float(
                nll_drift.max()
            ),

        "fp32_time_s":
            fp32["time_s"],
        "fp16_time_s":
            fp16["time_s"],
    }

    out_dir = Path("results")
    out_dir.mkdir(
        exist_ok=True
    )

    out = (
        out_dir
        / (
            f"{args.model}_"
            f"text_precision_quality_"
            f"{args.tag}.csv"
        )
    )

    with out.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                list(
                    summary.keys()
                )
            ),
        )

        writer.writeheader()
        writer.writerow(
            summary
        )

    print(
        "\nTEXT QUALITY SUMMARY"
    )
    print("-" * 72)

    for k, v in summary.items():
        print(
            f"{k:32s}: {v}"
        )

    print(
        "\nSaved:",
        out,
    )


if __name__ == "__main__":
    main()
