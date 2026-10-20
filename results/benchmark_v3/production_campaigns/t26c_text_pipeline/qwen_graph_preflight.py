import gc
import hashlib
import json
import os
import traceback
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("results/benchmark_v3").resolve()
OUT = BASE / "production_campaigns/t26c_text_pipeline"
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)
HUB = Path.home() / ".cache/huggingface/hub"

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

assert os.environ.get("HF_HUB_OFFLINE") == "1"
assert CORPUS.is_file()
torch.set_num_threads(4)
torch.manual_seed(2026)

texts = pq.read_table(
    CORPUS, columns=["text"]
)["text"].to_pylist()

excerpt = "\n".join(
    t.strip() for t in texts
    if isinstance(t, str) and t.strip()
)[:4000]

results = {}

for name, (repo, revision) in MODELS.items():
    folder = HUB / repo / "snapshots" / revision
    assert (folder / "config.json").is_file(), folder

    print("\nMODEL:", name, flush=True)
    print("LOADING FP32 CPU MODEL", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False,
    )

    token_ids = tokenizer.encode(
        excerpt,
        add_special_tokens=False,
    )
    assert len(token_ids) >= 128
    ids = torch.tensor(
        [token_ids[:128]], dtype=torch.long
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval().cpu()

    backbone = model.model
    layers = list(backbone.layers)
    assert len(layers) == 28

    positions = torch.arange(
        ids.shape[1], dtype=torch.long
    ).unsqueeze(0)

    # Causal mask: future positions cannot be attended.
    causal = torch.triu(
        torch.ones(
            ids.shape[1],
            ids.shape[1],
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    mask = torch.zeros(
        1, 1, ids.shape[1], ids.shape[1],
        dtype=torch.float32,
    ).masked_fill(
        causal,
        torch.finfo(torch.float32).min,
    )

    with torch.inference_mode():
        print("FULL MODEL FORWARD", flush=True)
        reference = model(
            input_ids=ids,
            use_cache=False,
            return_dict=True,
        ).logits.detach()

        assert bool(torch.isfinite(reference).all())

        hidden = backbone.embed_tokens(ids)

        position_embeddings = backbone.rotary_emb(
            hidden, positions
        )

        print("MANUAL 28-LAYER FORWARD", flush=True)
        for index, layer in enumerate(layers):
            hidden = layer(
                hidden_states=hidden,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]

            assert isinstance(hidden, torch.Tensor), (
                name, index, type(hidden)
            )
            assert bool(torch.isfinite(hidden).all()), (
                "NONFINITE_LAYER", name, index
            )

        actual = model.lm_head(
            backbone.norm(hidden)
        ).detach()

        assert actual.shape == reference.shape
        assert bool(torch.isfinite(actual).all())

        max_abs = float(
            (actual - reference).abs().max()
        )
        allclose = bool(torch.allclose(
            actual, reference,
            atol=1e-4, rtol=1e-4,
        ))
        agreement = float(
            (
                actual.argmax(dim=-1)
                == reference.argmax(dim=-1)
            ).float().mean()
        )

    print("MAX ABS ERROR:", max_abs, flush=True)
    print("TOKEN AGREEMENT:", agreement, flush=True)
    print(
        "CPU GRAPH:",
        "PASS" if allclose else "MISMATCH",
        flush=True,
    )

    results[name] = {
        "model_type": model.config.model_type,
        "layers": len(layers),
        "context": 128,
        "precision": "fp32",
        "max_abs_error": max_abs,
        "allclose": allclose,
        "token_agreement": agreement,
        "input_sha256": hashlib.sha256(
            ids.numpy().tobytes()
        ).hexdigest(),
    }

    del model, backbone, layers
    del reference, actual, hidden
    gc.collect()

destination = OUT / "qwen_graph_preflight_v1.json"
assert not destination.exists(), (
    "Previous preflight exists; refusing overwrite"
)
destination.write_text(
    json.dumps(results, indent=2) + "\n",
    encoding="utf-8",
)

if not all(x["allclose"] for x in results.values()):
    raise SystemExit(
        "QWEN GRAPH MISMATCH — NO PIPELINE "
        "MEASUREMENTS AUTHORIZED"
    )

print("\nQWEN2.5 + QWEN3 GRAPH PREFLIGHT: PASS")
print("DATABASE MODIFIED: NO")
print("PIPELINE PERFORMANCE MEASURED: NO")
print("RESULT:", destination)
