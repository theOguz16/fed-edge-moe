import gc
import hashlib
import json
import math
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path("results/benchmark_v3").resolve()
CAMPAIGN = BASE / "production_campaigns/t25_text_mac"
DIAG = CAMPAIGN / "diagnostics"
DB = BASE / (
    "production_campaigns/t23_mac_v1/"
    "dataset_v3_t23_mac.sqlite"
)
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)
MANIFEST = CAMPAIGN / "model_provenance.json"

TARGETS = [
    ("distilgpt2", "bf16"),
    ("qwen2_5_1_5b", "bf16"),
    ("qwen3_1_7b", "bf16"),
    ("qwen3_1_7b", "fp16"),
]
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def finite_report(x):
    mask = torch.isfinite(x)
    return {
        "total": x.numel(),
        "nonfinite": x.numel() - int(mask.sum().item()),
    }

def load_model(folder):
    return AutoModelForCausalLM.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

def inputs_for(model_name, folder, text, db):
    tokenizer = AutoTokenizer.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokens = tokenizer.encode(
        text, add_special_tokens=False
    )
    assert len(tokens) >= 128

    x = torch.tensor(
        [tokens[:128]], dtype=torch.long
    )
    digest = hashlib.sha256(
        x.numpy().tobytes()
    ).hexdigest()

    row = db.execute("""
        SELECT r.result_path
        FROM t25_results r
        JOIN t25_cases c ON c.case_id=r.case_id
        WHERE c.model=?
          AND c.precision='fp32'
          AND c.placement='full_cpu'
          AND c.qos='background'
          AND c.batch_size=1
    """, (model_name,)).fetchone()

    assert row is not None, model_name
    previous = json.loads(Path(row[0]).read_text())
    assert previous["input_tokens_sha256"] == digest, (
        "T25 INPUT TOKEN MISMATCH", model_name
    )
    return x, digest

def trace_qwen_fp16(folder, inputs):
    print("\n=== QWEN2.5 FP16 O_PROJ INPUT TRACE ===",
          flush=True)

    model = load_model(folder)
    model.to(device="cpu", dtype=torch.float16)

    proj = model.model.layers[0].self_attn.o_proj
    observations = {}

    def before(_module, args):
        observations["o_proj_input"] = finite_report(
            args[0]
        )

    def after(_module, _args, result):
        observations["o_proj_output"] = finite_report(
            result
        )

    a = proj.register_forward_pre_hook(before)
    b = proj.register_forward_hook(after)

    try:
        with torch.inference_mode():
            logits = model(
                input_ids=inputs,
                use_cache=False,
                return_dict=True,
            ).logits
        observations["final_logits"] = finite_report(
            logits
        )
    finally:
        a.remove()
        b.remove()
        del model
        gc.collect()

    assert "o_proj_input" in observations
    assert "o_proj_output" in observations

    for key, value in observations.items():
        print(key, json.dumps(value), flush=True)

    if observations["o_proj_input"]["nonfinite"]:
        conclusion = "NONFINITE_ALREADY_BEFORE_O_PROJ"
    elif observations["o_proj_output"]["nonfinite"]:
        conclusion = "NONFINITE_CREATED_IN_O_PROJ"
    else:
        conclusion = "O_PROJ_FINITE_IN_THIS_RUN"

    print("TRACE_CONCLUSION:", conclusion, flush=True)
    return {"observations": observations,
            "conclusion": conclusion}

def compare(model_name, precision, folder, inputs):
    dtype = DTYPES[precision]
    print(
        f"\n=== KL {model_name} {precision} ===",
        flush=True
    )

    model = load_model(folder)
    with torch.inference_mode():
        reference = model(
            input_ids=inputs,
            use_cache=False,
            return_dict=True,
        ).logits.detach().float().cpu()

    assert torch.isfinite(reference).all().item()
    model.to(device="cpu", dtype=dtype)

    output = []

    for device in ("cpu", "mps"):
        if device == "mps":
            model.to(device="mps", dtype=dtype)

        with torch.inference_mode():
            current = model(
                input_ids=inputs.to(device),
                use_cache=False,
                return_dict=True,
            ).logits.detach().float().cpu()

        if device == "mps":
            torch.mps.synchronize()

        if not torch.isfinite(current).all().item():
            entry = {
                "device": device,
                "status": "numerical_invalid",
                "finite": finite_report(current),
            }
        else:
            # KL(P_FP32 || P_low_precision).
            # All operations are on CPU in float32.
            with torch.inference_mode():
                lp = torch.log_softmax(
                    reference, dim=-1
                )
                lq = torch.log_softmax(
                    current, dim=-1
                )
                p = lp.exp()
                per_position = (
                    p * (lp - lq)
                ).sum(dim=-1)

            agreement = float(
                (
                    reference.argmax(dim=-1)
                    == current.argmax(dim=-1)
                ).float().mean().item()
            )

            entry = {
                "device": device,
                "status": "finite",
                "kl_direction": "FP32_to_low_precision",
                "mean_kl_nats": float(
                    per_position.mean().item()
                ),
                "max_kl_nats": float(
                    per_position.max().item()
                ),
                "token_agreement": agreement,
                "max_abs_logit_error": float(
                    (reference - current)
                    .abs().max().item()
                ),
                "allclose_atol_rtol_0_05": bool(
                    torch.allclose(
                        reference, current,
                        atol=0.05, rtol=0.05,
                    )
                ),
                "positions": int(
                    per_position.numel()
                ),
            }

            assert all(
                math.isfinite(v)
                for k, v in entry.items()
                if k in (
                    "mean_kl_nats",
                    "max_kl_nats",
                    "token_agreement",
                    "max_abs_logit_error",
                )
            )

        output.append(entry)
        print(json.dumps(entry), flush=True)

        del current
        gc.collect()

    del model, reference
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return output

def main():
    assert DB.is_file()
    assert CORPUS.is_file()
    assert MANIFEST.is_file()
    assert torch.backends.mps.is_available()

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    manifest = json.loads(MANIFEST.read_text())
    assert sha(CORPUS) == manifest["corpus_sha256"]

    db = sqlite3.connect(
        f"file:{DB}?mode=ro", uri=True
    )

    try:
        assert db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        raw = pq.read_table(
            CORPUS, columns=["text"]
        )["text"].to_pylist()
        excerpt = "\n".join(
            line.strip()
            for line in raw
            if isinstance(line, str) and line.strip()
        )[:4000]

        results = {
            "protocol": "T25_4_TARGETED_KL_V1",
            "timestamp_utc":
                datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": sha(CORPUS),
            "torch_version": torch.__version__,
            "context_length": 128,
            "batch_size": 1,
            "reference": "same_checkpoint_full_cpu_fp32",
            "kl_units": "nats_per_position",
            "qos": "not_controlled_diagnostic",
            "cases": [],
            "errors": [],
        }

        qwen_folder = Path(
            manifest["models"]["qwen2_5_1_5b"][
                "snapshot"
            ]
        )
        qwen_input, _ = inputs_for(
            "qwen2_5_1_5b",
            qwen_folder, excerpt, db
        )
        results["qwen_fp16_trace"] = (
            trace_qwen_fp16(
                qwen_folder, qwen_input
            )
        )

        for model_name, precision in TARGETS:
            folder = Path(
                manifest["models"][model_name][
                    "snapshot"
                ]
            )
            inputs, input_sha = inputs_for(
                model_name, folder, excerpt, db
            )

            try:
                values = compare(
                    model_name, precision,
                    folder, inputs
                )
                results["cases"].append({
                    "model": model_name,
                    "precision": precision,
                    "input_tokens_sha256":
                        input_sha,
                    "measurements": values,
                })
            except Exception as exc:
                results["errors"].append({
                    "model": model_name,
                    "precision": precision,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    "COMPARISON_FAILED:",
                    model_name, precision,
                    repr(exc),
                    flush=True
                )

        result_file = DIAG / "t25_4_results.json"
        result_file.write_text(
            json.dumps(
                results, indent=2,
                ensure_ascii=False
            ) + "\n"
        )

        print("\n=== T25.4 SUMMARY ===")
        print(
            "COMPLETED PRECISION GROUPS:",
            len(results["cases"]), "/4"
        )
        print(
            "COMPARISON ERRORS:",
            len(results["errors"])
        )
        print("RESULT FILE:", result_file)
        print("DATABASE MODIFIED: NO")

    finally:
        db.close()

if __name__ == "__main__":
    main()
