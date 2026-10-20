import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
OUT = BASE / "production_campaigns/g2_cut_graph_mac/text_pilot_v1"
CORPUS = BASE / (
    "text_quality_recovery/t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)

SNAPSHOTS = {
    "distilgpt2": (
        "models--distilgpt2",
        "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        1,
    ),
    "qwen2_5_1_5b": (
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        10,
    ),
    "qwen3_1_7b": (
        "models--Qwen--Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        10,
    ),
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + "\n")

def child(name, precision, path):
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.backends.mps.is_available()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    torch.set_num_threads(4)
    torch.manual_seed(2026)

    repo, revision, cut = SNAPSHOTS[name]
    snapshot = (
        Path.home() / ".cache/huggingface/hub"
        / repo / "snapshots" / revision
    )
    assert (snapshot / "config.json").is_file()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )

    texts = pq.read_table(
        CORPUS, columns=["text"]
    )["text"].to_pylist()

    excerpt = "\n".join(
        t.strip() for t in texts
        if isinstance(t, str) and t.strip()
    )[:4000]

    tokens = tokenizer.encode(
        excerpt, add_special_tokens=False
    )
    assert len(tokens) >= 256

    ids = torch.tensor(
        [tokens[:128], tokens[128:256]],
        dtype=torch.long,
    )
    inputs_sha = hashlib.sha256(
        ids.numpy().tobytes()
    ).hexdigest()

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval().cpu()

    with torch.inference_mode():
        ref = model(
            input_ids=ids,
            use_cache=False,
            return_dict=True,
        ).logits.detach().cpu()

    assert bool(torch.isfinite(ref).all())

    if name == "distilgpt2":
        backbone = model.transformer
        blocks = list(backbone.h)
        embedding = backbone.wte
    else:
        backbone = model.model
        blocks = list(backbone.layers)
        embedding = backbone.embed_tokens

    head = model.lm_head
    assert embedding.weight.data_ptr() == head.weight.data_ptr()

    # Preserve tied-weight semantics without forcing
    # the embedding and output head onto the same device.
    head.weight = torch.nn.Parameter(
        head.weight.detach().clone(),
        requires_grad=False,
    )

    dtype = (
        torch.float16
        if precision == "fp16"
        else torch.float32
    )

    first, second = "cpu", "mps"

    for i, layer in enumerate(blocks):
        layer.to(
            device=first if i < cut else second,
            dtype=dtype,
        )

    if name == "distilgpt2":
        backbone.wte.to(device=first, dtype=dtype)
        backbone.wpe.to(device=first, dtype=dtype)
        backbone.drop.to(device=first, dtype=dtype)
        backbone.ln_f.to(device=second, dtype=dtype)
    else:
        backbone.embed_tokens.to(device=first, dtype=dtype)
        backbone.norm.to(device=second, dtype=dtype)

    head.to(device=second, dtype=dtype)

    positions = {
        d: torch.arange(
            128, dtype=torch.long, device=d
        ).unsqueeze(0)
        for d in ("cpu", "mps")
    }

    future = torch.triu(
        torch.ones(128, 128, dtype=torch.bool),
        diagonal=1,
    )
    mask_cpu = torch.zeros(
        (1, 1, 128, 128), dtype=torch.float32
    ).masked_fill(
        future, torch.finfo(torch.float32).min
    )
    masks = {
        "cpu": mask_cpu,
        "mps": mask_cpu.to("mps"),
    }

    rotary = {}
    if name != "distilgpt2":
        for device in ("cpu", "mps"):
            dummy = torch.zeros(
                1, 128, model.config.hidden_size,
                dtype=dtype, device=device,
            )
            with torch.inference_mode():
                rotary[device] = backbone.rotary_emb(
                    dummy, positions[device]
                )

    def blocks_forward(hidden, selected):
        dev = hidden.device.type

        for layer in selected:
            if name == "distilgpt2":
                result = layer(
                    hidden,
                    attention_mask=masks[dev],
                    use_cache=False,
                )
            else:
                result = layer(
                    hidden_states=hidden,
                    attention_mask=masks[dev],
                    position_ids=positions[dev],
                    position_embeddings=rotary[dev],
                    use_cache=False,
                )

            hidden = (
                result[0]
                if isinstance(result, (tuple, list))
                else result
            )

        return hidden

    with torch.inference_mode():
        if name == "distilgpt2":
            hidden = backbone.drop(
                backbone.wte(ids)
                + backbone.wpe(positions["cpu"])
            )
        else:
            hidden = backbone.embed_tokens(ids)


        print(
            "EMBEDDING finite=",
            bool(torch.isfinite(hidden).all()),
            "dtype=", hidden.dtype,
            "max_abs_finite=",
            float(hidden[torch.isfinite(hidden)].abs().max())
            if bool(torch.isfinite(hidden).any()) else None,
            flush=True,
        )
        print(
            "MASK finite=", bool(torch.isfinite(masks["cpu"]).all()),
            "ROTARY finite=",
            all(bool(torch.isfinite(x).all())
                for x in rotary["cpu"]),
            flush=True,
        )
        first_layer = blocks[0]
        print(
            "BLOCK0 WEIGHTS finite=",
            all(bool(torch.isfinite(p).all())
                for p in first_layer.parameters()),
            flush=True,
        )
        for block_index, layer in enumerate(blocks[:cut]):
            hidden = blocks_forward(hidden, [layer])
            finite = bool(torch.isfinite(hidden).all())
            print(
                'TRACE CPU', block_index,
                'finite=', finite,
                'shape=', tuple(hidden.shape),
                flush=True,
            )
        activation_bytes = (
            hidden.numel() * hidden.element_size()
        )
        hidden_shape = list(hidden.shape)

        hidden = hidden.to("mps")
        for block_index, layer in enumerate(blocks[cut:], start=cut):
            hidden = blocks_forward(hidden, [layer])
            finite = bool(torch.isfinite(hidden).all())
            print(
                'TRACE MPS', block_index,
                'finite=', finite,
                'shape=', tuple(hidden.shape),
                flush=True,
            )

        if name == "distilgpt2":
            actual = head(backbone.ln_f(hidden))
        else:
            actual = head(backbone.norm(hidden))

        torch.mps.synchronize()
        actual = actual.float().cpu()

    shape_ok = actual.shape == ref.shape
    finite = bool(torch.isfinite(actual).all())
    if not shape_ok or not finite:
        save(path, {
            "model": name,
            "precision": precision,
            "cut": cut,
            "shape_match": shape_ok,
            "finite": finite,
            "status": "numerical_invalid",
            "input_sha256": inputs_sha,
            "activation_bytes": activation_bytes,
            "activation_shape": hidden_shape,
        })
        return

    diff = (actual - ref).abs()
    max_abs = float(diff.max())
    token_agreement = float(
        (actual.argmax(-1) == ref.argmax(-1))
        .float().mean()
    )
    strict = bool(torch.allclose(
        actual, ref, atol=1e-3, rtol=1e-3
    ))

    # KL(P_fp32 || P_candidate), averaged per position.
    p = torch.log_softmax(ref.float(), dim=-1)
    q = torch.log_softmax(actual.float(), dim=-1)
    kl = float(
        (p.exp() * (p - q)).sum(dim=-1).mean()
    )

    save(path, {
        "protocol": "G2_TEXT_NUMERICAL_PILOT_V1",
        "model": name,
        "precision": precision,
        "cut": cut,
        "direction": "cpu_mps",
        "reference": "full_cpu_fp32",
        "snapshot_revision": revision,
        "input_sha256": inputs_sha,
        "context_tokens": 128,
        "batch_size": 2,
        "activation_shape": hidden_shape,
        "activation_bytes": activation_bytes,
        "shape_match": shape_ok,
        "finite": finite,
        "max_abs_logit_error": max_abs,
        "token_agreement": token_agreement,
        "strict_allclose_1e_3": strict,
        "kl_fp32_to_candidate_nats": kl,
        "status": (
            "numerical_pass"
            if strict and token_agreement == 1.0
            else "numerical_mismatch"
        ),
        "pipeline_concurrency_measured": False,
        "latency_measured": False,
        "power_measured": False,
        "quality_generalization_established": False,
    })

def parent():
    assert CORPUS.is_file()
    assert not OUT.exists(), (
        "Previous pilot exists; refusing overwrite"
    )
    OUT.mkdir(parents=True)

    summary = []
    for name in SNAPSHOTS:
        for precision in ("fp32", "fp16"):
            label = name + "_" + precision
            dest = OUT / (label + ".json")
            stderr = OUT / (label + ".stderr.txt")
            stdout = OUT / (label + ".stdout.txt")

            print("RUN:", label, flush=True)

            with stderr.open("w") as err, \
                 stdout.open("w") as out:
                try:
                    proc = subprocess.run(
                        [
                            sys.executable, "-u",
                            str(Path(__file__).resolve()),
                            "child", name, precision, str(dest),
                        ],
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                        },
                        stdout=out,
                        stderr=err,
                        timeout=900,
                    )
                    code = proc.returncode
                except subprocess.TimeoutExpired:
                    code = 124

            if code == 0 and dest.is_file():
                record = json.loads(dest.read_text())
                status = record["status"]
                agreement = record.get("token_agreement")
                kl = record.get("kl_fp32_to_candidate_nats")
            else:
                status = "execution_failed"
                agreement = None
                kl = None

            item = {
                "model": name,
                "precision": precision,
                "status": status,
                "token_agreement": agreement,
                "kl_nats": kl,
                "exit_code": code,
                "result_path": str(dest) if dest.is_file() else None,
                "result_sha256": sha(dest) if dest.is_file() else None,
                "stderr_tail": stderr.read_text(
                    errors="replace"
                )[-900:] if status == "execution_failed" else "",
            }
            summary.append(item)
            print(
                "STATUS:", status,
                "TOKEN_AGREEMENT:", agreement,
                "KL:", kl,
                flush=True,
            )
            if status == "execution_failed":
                print("ERROR:", item["stderr_tail"][-350:])

    save(OUT / "summary.json", {
        "protocol": "G2_TEXT_NUMERICAL_PILOT_V1",
        "source_sha256": sha(Path(__file__).resolve()),
        "cases": summary,
        "notes": [
            "Two 128-token WikiText-2 contexts",
            "CPU-to-MPS split forward; not concurrent pipeline",
            "FP32 reference is full CPU",
            "KL is a diagnostic on two contexts, not a corpus quality score",
            "No throughput or power measurement",
        ],
    })

    print("\nTEXT PILOT: FINISHED")
    print("OUTPUT:", OUT)
    print("SQLITE MODIFIED: NO")
    print("OLD BENCHMARKS MODIFIED: NO")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        parent()
    else:
        assert sys.argv[1] == "child"
        try:
            child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
        except BaseException:
            traceback.print_exc()
            raise
