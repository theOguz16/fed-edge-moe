import gc
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("results/benchmark_v3").resolve()
ROOT = BASE / "production_campaigns/t26c_text_pipeline"
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)
HUB = Path.home() / ".cache/huggingface/hub"
OUT = ROOT / "qwen_hybrid_preflight_v2.json"

MODELS = {
    "qwen2_5_1_5b": (
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    ),
    "qwen3_1_7b": (
        "models--Qwen--Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    ),
}

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()
assert CORPUS.is_file()
assert not OUT.exists(), "Existing result: refusing overwrite"

torch.set_num_threads(4)
torch.manual_seed(2026)

texts = pq.read_table(
    CORPUS, columns=["text"]
)["text"].to_pylist()
excerpt = "\n".join(
    x.strip() for x in texts
    if isinstance(x, str) and x.strip()
)[:4000]

results = []

for name, (repository, revision) in MODELS.items():
    print("\nMODEL:", name, flush=True)

    snapshot = HUB / repository / "snapshots" / revision
    assert (snapshot / "config.json").is_file()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokens = tokenizer.encode(
        excerpt, add_special_tokens=False
    )
    assert len(tokens) >= 128

    ids = torch.tensor(
        [tokens[:128]], dtype=torch.long
    )
    input_sha = hashlib.sha256(
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

    backbone = model.model
    layers = list(backbone.layers)
    assert len(layers) == 28

    positions_cpu = torch.arange(
        128, dtype=torch.long
    ).unsqueeze(0)

    forbidden = torch.triu(
        torch.ones(128, 128, dtype=torch.bool),
        diagonal=1,
    )
    mask_cpu = torch.zeros(
        (1, 1, 128, 128),
        dtype=torch.float32,
    ).masked_fill(
        forbidden, torch.finfo(torch.float32).min
    )

    with torch.inference_mode():
        reference = model(
            input_ids=ids,
            use_cache=False,
            return_dict=True,
        ).logits.detach().clone()

    assert torch.isfinite(reference).all().item()

    embedding = backbone.embed_tokens
    head = model.lm_head

    assert (
        embedding.weight is head.weight
        or embedding.weight.data_ptr()
           == head.weight.data_ptr()
    ), "Unexpected untied weights"

    # Separate output weights without changing their values.
    head.weight = torch.nn.Parameter(
        head.weight.detach().clone(),
        requires_grad=False,
    )
    assert (
        embedding.weight.data_ptr()
        != head.weight.data_ptr()
    ), "Weight separation failed"

    print("TIED WEIGHTS SAFELY SEPARATED", flush=True)

    for direction in ("cpu_mps", "mps_cpu"):
        first, second = (
            ("cpu", "mps")
            if direction == "cpu_mps"
            else ("mps", "cpu")
        )

        embedding.to(first)
        for index, layer in enumerate(layers):
            layer.to(first if index < 14 else second)

        backbone.norm.to(second)
        head.to(second)
        torch.mps.synchronize()

        assert embedding.weight.device.type == first
        assert head.weight.device.type == second

        # Cache model-independent tensors per device.
        positions = {
            "cpu": positions_cpu,
            "mps": positions_cpu.to("mps"),
        }
        masks = {
            "cpu": mask_cpu,
            "mps": mask_cpu.to("mps"),
        }

        with torch.inference_mode():
            hidden = embedding(ids.to(first))

            for index, layer in enumerate(layers):
                device = (
                    first if index < 14 else second
                )
                assert next(
                    layer.parameters()
                ).device.type == device

                if hidden.device.type != device:
                    hidden = hidden.to(device)
                    torch.mps.synchronize()

                rotary = backbone.rotary_emb(
                    hidden, positions[device]
                )

                hidden = layer(
                    hidden_states=hidden,
                    attention_mask=masks[device],
                    position_ids=positions[device],
                    position_embeddings=rotary,
                    use_cache=False,
                )

                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]

                assert isinstance(hidden, torch.Tensor)
                assert torch.isfinite(hidden).all().item(), (
                    "NONFINITE", name, direction, index
                )

            actual = head(
                backbone.norm(hidden)
            ).to("cpu")
            torch.mps.synchronize()

            assert torch.isfinite(actual).all().item()

            difference = (
                actual - reference
            ).abs()

            max_abs = float(difference.max().item())
            agreement = float(
                (
                    actual.argmax(dim=-1)
                    == reference.argmax(dim=-1)
                ).float().mean().item()
            )
            allclose = bool(torch.allclose(
                actual, reference,
                atol=1e-3, rtol=1e-3,
            ))

        print(
            direction,
            f"max_abs={max_abs:.7f}",
            f"token_agreement={agreement:.4f}",
            "NUMERICAL_PASS" if allclose
            else "NUMERICAL_MISMATCH",
            flush=True,
        )

        results.append({
            "model": name,
            "direction": direction,
            "cut_after_layer": 14,
            "precision": "fp32",
            "context_tokens": 128,
            "input_sha256": input_sha,
            "max_abs_error": max_abs,
            "token_agreement": agreement,
            "numerical_allclose": allclose,
            "tied_weights_detached": True,
        })

    del model, backbone, layers
    del embedding, head, hidden, actual, reference
    gc.collect()
    torch.mps.empty_cache()

ROOT.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(results, indent=2) + "\n",
    encoding="utf-8",
)

passed = sum(x["numerical_allclose"] for x in results)
print(f"\nQWEN HYBRID FP32 NUMERICAL: {passed}/4 PASS")
print("RESULT:", OUT)
print("DATABASE MODIFIED: NO")
print("POWER MEASURED: NO")
print("TWO-THREAD PIPELINE MEASURED: NO")

if passed != 4:
    raise SystemExit(
        "NUMERICAL MISMATCH: "
        "configuration requires investigation"
    )
