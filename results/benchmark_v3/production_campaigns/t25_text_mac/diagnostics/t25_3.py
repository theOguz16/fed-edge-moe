import json
import re
import sqlite3
import statistics
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "results/benchmark_v3"
DB = BASE / "production_campaigns/t23_mac_v1/dataset_v3_t23_mac.sqlite"
CAMPAIGN = BASE / "production_campaigns/t25_text_mac"
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)

def audit_power():
    print("\n=== LOW POWER RAW SENSOR AUDIT ===", flush=True)
    db = sqlite3.connect(
        f"file:{DB.resolve()}?mode=ro", uri=True
    )
    try:
        rows = db.execute("""
            SELECT c.model,c.precision,c.placement,
                   c.qos,c.batch_size,
                   r.result_path,r.power_sample_count,
                   r.cpu_power_w+r.gpu_power_w,
                   r.elapsed_s
            FROM t25_results r
            JOIN t25_cases c ON c.case_id=r.case_id
            ORDER BY r.cpu_power_w+r.gpu_power_w
            LIMIT 8
        """).fetchall()
    finally:
        db.close()

    for row in rows:
        model, precision, placement, qos, batch, \
            result_path, expected, stored_w, elapsed = row

        power_file = Path(result_path).with_name(
            "powermetrics_raw.txt"
        )
        assert power_file.is_file(), power_file
        raw = power_file.read_text(errors="replace")

        cpu = list(re.finditer(
            r"CPU Power:\s*([\d.]+)\s*mW",
            raw, re.I
        ))
        pairs = []
        for i, match in enumerate(cpu):
            stop = (
                cpu[i+1].start()
                if i+1 < len(cpu) else len(raw)
            )
            gpu = re.search(
                r"GPU Power:\s*([\d.]+)\s*mW",
                raw[match.end():stop], re.I
            )
            if gpu:
                pairs.append((
                    float(match.group(1))/1000,
                    float(gpu.group(1))/1000
                ))

        assert len(pairs) == expected, (
            "Power sample count changed", result_path
        )
        powers = [a+b for a,b in pairs]
        assert powers
        calculated = statistics.mean(powers)
        assert abs(calculated-stored_w) < 1e-6

        print(
            f"{model} {precision} {placement} "
            f"{qos} batch={batch}",
            f"samples={len(pairs)}",
            f"duration={elapsed:.1f}s",
            f"mean={calculated:.4f}W",
            f"median={statistics.median(powers):.4f}W",
            f"range=[{min(powers):.4f},"
            f"{max(powers):.4f}]W",
            flush=True
        )

    print("RAW POWER CONSISTENCY: PASS", flush=True)

class FoundNonfinite(Exception):
    pass

def tensor_of(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = tensor_of(item)
            if result is not None:
                return result
    if isinstance(value, dict):
        for item in value.values():
            result = tensor_of(item)
            if result is not None:
                return result
    return None

def inspect_first_layer():
    import torch
    import pyarrow.parquet as pq
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM
    )

    print("\n=== QWEN2.5 FP16 LAYER-0 TRACE ===", flush=True)

    torch.set_num_threads(4)
    manifest = json.loads(
        (CAMPAIGN / "model_provenance.json").read_text()
    )
    folder = Path(
        manifest["models"]["qwen2_5_1_5b"]["snapshot"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False
    )

    raw = pq.read_table(
        CORPUS, columns=["text"]
    )["text"].to_pylist()
    excerpt = "\n".join(
        line.strip() for line in raw
        if isinstance(line, str) and line.strip()
    )[:4000]

    tokens = tokenizer.encode(
        excerpt, add_special_tokens=False
    )
    assert len(tokens) >= 128

    inputs = torch.tensor(
        [tokens[:128]], dtype=torch.long
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True
    ).eval()

    model.to(device="cpu", dtype=torch.float16)
    layer = model.model.layers[0]

    for name, parameter in layer.named_parameters():
        valid = torch.isfinite(parameter)
        if not valid.all().item():
            print(
                "NONFINITE_PARAMETER:",
                name,
                "bad=", parameter.numel() -
                    int(valid.sum().item()),
                flush=True
            )
            return

    print("LAYER_0_PARAMETERS: FINITE", flush=True)

    visited = []
    hooks = []

    def monitor(name):
        def check(_module, _args, output):
            tensor = tensor_of(output)
            if tensor is None:
                return
            finite = torch.isfinite(tensor)
            bad = tensor.numel() - int(
                finite.sum().item()
            )
            visited.append(name)
            if bad:
                print(
                    "FIRST_NONFINITE_SUBMODULE:",
                    name,
                    flush=True
                )
                print(
                    "BAD_VALUES:",
                    bad, "/", tensor.numel(),
                    flush=True
                )
                print(
                    "SHAPE:", tuple(tensor.shape),
                    flush=True
                )
                raise FoundNonfinite(name)
        return check

    hooks.append(
        model.model.embed_tokens.register_forward_hook(
            monitor("embedding")
        )
    )

    for name, module in layer.named_modules():
        if name and not list(module.children()):
            hooks.append(
                module.register_forward_hook(
                    monitor("layer_0." + name)
                )
            )

    hooks.append(
        layer.register_forward_hook(
            monitor("layer_0.output")
        )
    )

    try:
        with torch.inference_mode():
            result = model(
                input_ids=inputs,
                use_cache=False,
                return_dict=True
            ).logits

        print(
            "FORWARD_FINITE:",
            bool(torch.isfinite(result).all().item()),
            flush=True
        )
    except FoundNonfinite:
        print(
            "LAST_CHECKED_MODULES:",
            visited[-8:],
            flush=True
        )
    finally:
        for hook in hooks:
            hook.remove()
        del model

    print("SQLITE MODIFIED: NO", flush=True)

if __name__ == "__main__":
    audit_power()
    inspect_first_layer()
