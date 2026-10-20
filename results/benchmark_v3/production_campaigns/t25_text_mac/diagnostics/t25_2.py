import json
import math
import sqlite3
import sys
from pathlib import Path

import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path.cwd()
BASE = ROOT / "results/benchmark_v3"
DB = BASE / "production_campaigns/t23_mac_v1/dataset_v3_t23_mac.sqlite"
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)
MANIFEST = BASE / (
    "production_campaigns/t25_text_mac/model_provenance.json"
)

def audit_database():
    print("\n=== T25.2 DATABASE AUDIT ===", flush=True)
    uri = f"file:{DB.resolve()}?mode=ro"
    db = sqlite3.connect(uri, uri=True)

    try:
        print("integrity:", db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0])

        for row in db.execute("""
            SELECT status, COUNT(*)
            FROM t25_cases
            GROUP BY status
            ORDER BY status
        """):
            print("case_status:", *row)

        print("\nNUMERICAL STATUS:")
        for row in db.execute("""
            SELECT c.model, c.precision,
                   r.numerical_status, COUNT(*)
            FROM t25_results r
            JOIN t25_cases c
              ON c.case_id=r.case_id
            GROUP BY c.model,c.precision,
                     r.numerical_status
            ORDER BY c.model,c.precision,
                     r.numerical_status
        """):
            print(" | ".join(map(str, row)))

        print("\nLOWEST MEASURED SOC POWER:")
        for row in db.execute("""
            SELECT c.model,c.precision,
                   c.placement,c.qos,
                   c.batch_size,
                   ROUND(r.cpu_power_w+r.gpu_power_w,4),
                   r.power_sample_count,
                   ROUND(r.elapsed_s,2)
            FROM t25_results r
            JOIN t25_cases c
              ON c.case_id=r.case_id
            ORDER BY r.cpu_power_w+r.gpu_power_w
            LIMIT 12
        """):
            print(" | ".join(map(str, row)))

        print("\nQWEN2.5 FP16 FAILURE MESSAGES:")
        for row in db.execute("""
            SELECT c.placement,
                   COUNT(*),
                   COUNT(DISTINCT a.message)
            FROM t25_cases c
            JOIN t25_attempts a
              ON a.case_id=c.case_id
            WHERE c.model='qwen2_5_1_5b'
              AND c.precision='fp16'
              AND a.status='numerical_invalid'
            GROUP BY c.placement
            ORDER BY c.placement
        """):
            print(" | ".join(map(str, row)))

        print("\nT25 ROW COUNT:",
              db.execute(
                  "SELECT COUNT(*) FROM t25_results"
              ).fetchone()[0])

    finally:
        db.close()

class FirstNonfinite(Exception):
    pass

def first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for value in obj:
            found = first_tensor(value)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for value in obj.values():
            found = first_tensor(value)
            if found is not None:
                return found
    return None

def probe(device, snapshot, input_ids):
    print(f"\n=== QWEN2.5 FP16 {device.upper()} ===",
          flush=True)

    torch.set_num_threads(4)

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

    model.to(device=device, dtype=torch.float16)

    handles = []
    checked = []

    def monitor(name):
        def hook(_module, _args, output):
            tensor = first_tensor(output)
            if tensor is None:
                return

            total = tensor.numel()
            finite = torch.isfinite(tensor)
            bad = total - int(finite.sum().item())

            checked.append(name)

            if bad:
                print(
                    f"FIRST_NONFINITE_LAYER={name}",
                    f"bad_values={bad}/{total}",
                    f"shape={tuple(tensor.shape)}",
                    flush=True
                )
                raise FirstNonfinite(name)

        return hook

    modules = [
        ("embedding", model.model.embed_tokens),
        *[
            (f"decoder_layer_{i}", layer)
            for i, layer in enumerate(model.model.layers)
        ],
        ("final_norm", model.model.norm),
        ("lm_head", model.lm_head),
    ]

    for name, module in modules:
        handles.append(
            module.register_forward_hook(monitor(name))
        )

    try:
        with torch.inference_mode():
            result = model(
                input_ids=input_ids.to(device),
                use_cache=False,
                return_dict=True,
            ).logits

            if device == "mps":
                torch.mps.synchronize()

            if torch.isfinite(result).all().item():
                print(
                    "OUTPUT_FINITE: YES",
                    "layers_checked:", len(checked),
                    flush=True
                )
            else:
                print(
                    "OUTPUT_FINITE: NO",
                    "no hook captured first failure",
                    flush=True
                )

    except FirstNonfinite as exc:
        print(
            "DIAGNOSIS: nonfinite appeared at",
            str(exc),
            flush=True
        )
    except Exception as exc:
        print(
            "DIAGNOSTIC_RUNTIME_ERROR:",
            type(exc).__name__,
            str(exc)[:1000],
            flush=True
        )
    finally:
        for handle in handles:
            handle.remove()
        del model
        if device == "mps":
            torch.mps.empty_cache()

def main():
    audit_database()

    assert MANIFEST.is_file(), (
        "T25 provenance manifest not found"
    )

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )
    snapshot = Path(
        manifest["models"]["qwen2_5_1_5b"]["snapshot"]
    )

    assert snapshot.is_dir()
    assert CORPUS.is_file()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )

    raw = pq.read_table(
        CORPUS, columns=["text"]
    )["text"].to_pylist()

    excerpt = "\n".join(
        line.strip()
        for line in raw
        if isinstance(line, str) and line.strip()
    )[:4000]

    tokens = tokenizer.encode(
        excerpt, add_special_tokens=False
    )
    assert len(tokens) >= 256

    # Same first 128-token sequence as T25 batch=1.
    inputs = torch.tensor(
        [tokens[:128]], dtype=torch.long
    )

    print("\nINPUT TOKENS:", tuple(inputs.shape))
    print("TORCH:", torch.__version__)
    print("MPS AVAILABLE:",
          torch.backends.mps.is_available())

    probe("cpu", snapshot, inputs)

    if torch.backends.mps.is_available():
        probe("mps", snapshot, inputs)

    print("\nT25.2 DIAGNOSTIC FINISHED")
    print("DATABASE MODIFIED: NO")

if __name__ == "__main__":
    main()
