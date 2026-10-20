import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
G4 = BASE / "production_campaigns/g4_campaign_mac_v1"
OUT = G4 / "text_fp16_quality_v1"
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)

PROTOCOL = "G4_TEXT_FP16_QUALITY_V1"
MODELS = ("distilgpt2", "qwen3_1_7b")
DIRECTIONS = ("cpu_mps", "mps_cpu")
CONTEXT = 128
CONTEXT_COUNT = 8

SNAPSHOTS = {
    "distilgpt2": (
        "models--distilgpt2",
        "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        1,
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

def atomic_json(path, obj):
    assert not path.exists(), path
    temp = path.with_suffix(".tmp")
    assert not temp.exists(), temp
    with temp.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def evaluate(name, direction):
    import torch
    import torch.nn.functional as F
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    repo, revision, cut = SNAPSHOTS[name]
    snapshot = (
        Path.home() / ".cache/huggingface/hub"
        / repo / "snapshots" / revision
    )
    assert (snapshot / "config.json").is_file()
    assert CORPUS.is_file()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )

    texts = pq.read_table(
        CORPUS, columns=["text"]
    )["text"].to_pylist()

    excerpt = "\n".join(
        text.strip()
        for text in texts
        if isinstance(text, str) and text.strip()
    )[:30000]

    tokens = tokenizer.encode(
        excerpt, add_special_tokens=False
    )

    needed = CONTEXT * CONTEXT_COUNT
    assert len(tokens) >= needed, (
        "Not enough tokens", len(tokens), needed
    )

    ids = torch.tensor(
        [
            tokens[i * CONTEXT:(i + 1) * CONTEXT]
            for i in range(CONTEXT_COUNT)
        ],
        dtype=torch.long,
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

    # Full CPU FP32 reference, computed before partitioning.
    references = []
    with torch.inference_mode():
        for i in range(CONTEXT_COUNT):
            logits = model(
                input_ids=ids[i:i+1],
                use_cache=False,
                return_dict=True,
            ).logits.detach().float().cpu()

            assert logits.shape[:2] == (1, CONTEXT)
            assert bool(torch.isfinite(logits).all())
            references.append(logits.squeeze(0))

    distil = name == "distilgpt2"
    backbone = model.transformer if distil else model.model
    blocks = list(
        backbone.h if distil else backbone.layers
    )

    embedding = (
        backbone.wte if distil
        else backbone.embed_tokens
    )
    head = model.lm_head

    assert embedding.weight.data_ptr() == head.weight.data_ptr()

    # Preserve the same tied-weight separation used in G2/G4.
    head.weight = torch.nn.Parameter(
        head.weight.detach().clone(),
        requires_grad=False,
    )

    assert embedding.weight.data_ptr() != head.weight.data_ptr()

    first, second = (
        ("cpu", "mps")
        if direction == "cpu_mps"
        else ("mps", "cpu")
    )

    for index, block in enumerate(blocks):
        block.to(
            device=first if index < cut else second,
            dtype=torch.float16,
        )

    if distil:
        backbone.wte.to(first, dtype=torch.float16)
        backbone.wpe.to(first, dtype=torch.float16)
        backbone.drop.to(first, dtype=torch.float16)
        backbone.ln_f.to(second, dtype=torch.float16)
    else:
        backbone.embed_tokens.to(first, dtype=torch.float16)
        backbone.norm.to(second, dtype=torch.float16)

    head.to(second, dtype=torch.float16)

    positions = {
        device: torch.arange(
            CONTEXT, dtype=torch.long, device=device
        ).unsqueeze(0)
        for device in ("cpu", "mps")
    }

    future = torch.triu(
        torch.ones(
            CONTEXT, CONTEXT, dtype=torch.bool
        ),
        diagonal=1,
    )

    mask_cpu = torch.zeros(
        (1, 1, CONTEXT, CONTEXT),
        dtype=torch.float32,
    ).masked_fill(
        future,
        torch.finfo(torch.float32).min,
    )

    masks = {
        "cpu": mask_cpu,
        "mps": mask_cpu.to("mps"),
    }

    rotary = {}
    if not distil:
        for device in ("cpu", "mps"):
            dummy = torch.zeros(
                1, CONTEXT,
                model.config.hidden_size,
                dtype=torch.float16,
                device=device,
            )
            with torch.inference_mode():
                rotary[device] = backbone.rotary_emb(
                    dummy, positions[device]
                )
            del dummy

    def forward_blocks(hidden, selected):
        device = hidden.device.type

        for block in selected:
            if distil:
                result = block(
                    hidden,
                    attention_mask=masks[device],
                    use_cache=False,
                )
            else:
                result = block(
                    hidden_states=hidden,
                    attention_mask=masks[device],
                    position_ids=positions[device],
                    position_embeddings=rotary[device],
                    use_cache=False,
                )

            hidden = (
                result[0]
                if isinstance(result, (tuple, list))
                else result
            )

        return hidden

    total_positions = 0
    token_matches = 0
    ref_ce_sum = 0.0
    fp16_ce_sum = 0.0
    kl_sum = 0.0
    max_abs = 0.0
    invalid_contexts = 0
    per_context = []

    with torch.inference_mode():
        for i in range(CONTEXT_COUNT):
            sample = ids[i:i+1]
            ref = references[i]

            if distil:
                hidden = backbone.drop(
                    backbone.wte(sample.to(first))
                    + backbone.wpe(positions[first])
                )
            else:
                hidden = backbone.embed_tokens(
                    sample.to(first)
                )

            hidden = forward_blocks(
                hidden, blocks[:cut]
            )

            logical_bytes = (
                hidden.numel() * hidden.element_size()
            )

            hidden = hidden.to(second, copy=True)
            hidden = forward_blocks(
                hidden, blocks[cut:]
            )

            if distil:
                output = head(backbone.ln_f(hidden))
            else:
                output = head(backbone.norm(hidden))

            torch.mps.synchronize()
            candidate = output.squeeze(0).float().cpu()

            finite = bool(torch.isfinite(candidate).all())
            shape_ok = candidate.shape == ref.shape

            record = {
                "context_index": i,
                "finite": finite,
                "shape_match": shape_ok,
                "logical_activation_bytes": logical_bytes,
            }

            if not finite or not shape_ok:
                invalid_contexts += 1
                per_context.append(record)
                continue

            # Next-token labels: position t predicts token t+1.
            labels = sample[0, 1:].cpu()
            p_logits = ref[:-1]
            q_logits = candidate[:-1]

            n = len(labels)
            p_ce = F.cross_entropy(
                p_logits, labels, reduction="sum"
            ).item()
            q_ce = F.cross_entropy(
                q_logits, labels, reduction="sum"
            ).item()

            p_log = F.log_softmax(p_logits, dim=-1)
            q_log = F.log_softmax(q_logits, dim=-1)
            kl = (
                (p_log.exp() * (p_log - q_log))
                .sum(dim=-1)
                .sum()
                .item()
            )

            matches = int(
                (p_logits.argmax(-1) == q_logits.argmax(-1))
                .sum()
            )

            error = float(
                (candidate - ref).abs().max()
            )

            assert all(
                math.isfinite(x)
                for x in (p_ce, q_ce, kl, error)
            )

            total_positions += n
            token_matches += matches
            ref_ce_sum += p_ce
            fp16_ce_sum += q_ce
            kl_sum += kl
            max_abs = max(max_abs, error)

            record.update({
                "evaluated_next_token_positions": n,
                "token_matches": matches,
                "reference_cross_entropy": p_ce / n,
                "fp16_cross_entropy": q_ce / n,
                "kl_fp32_to_fp16_nats": kl / n,
                "max_abs_logit_error": error,
            })
            per_context.append(record)

            del output, candidate, hidden

    metrics = {
        "protocol": PROTOCOL,
        "model": name,
        "direction": direction,
        "precision": "fp16",
        "cut_after_block": cut,
        "context_count": CONTEXT_COUNT,
        "context_tokens": CONTEXT,
        "input_sha256": input_sha,
        "snapshot_revision": revision,
        "reference": "full_cpu_fp32",
        "attention_implementation": "eager",
        "use_cache": False,
        "invalid_context_count": invalid_contexts,
        "valid_next_token_positions": total_positions,
        "token_agreement": (
            token_matches / total_positions
            if total_positions else None
        ),
        "reference_cross_entropy": (
            ref_ce_sum / total_positions
            if total_positions else None
        ),
        "fp16_cross_entropy": (
            fp16_ce_sum / total_positions
            if total_positions else None
        ),
        "cross_entropy_delta": (
            (fp16_ce_sum - ref_ce_sum) / total_positions
            if total_positions else None
        ),
        "kl_fp32_to_fp16_nats": (
            kl_sum / total_positions
            if total_positions else None
        ),
        "reference_perplexity": (
            math.exp(ref_ce_sum / total_positions)
            if total_positions
            and ref_ce_sum / total_positions < 700
            else None
        ),
        "fp16_perplexity": (
            math.exp(fp16_ce_sum / total_positions)
            if total_positions
            and fp16_ce_sum / total_positions < 700
            else None
        ),
        "max_abs_logit_error": (
            max_abs if total_positions else None
        ),
        "contexts": per_context,
        "corpus_sha256": sha(CORPUS),
        "throughput_measured": False,
        "energy_measured": False,
        "generalization_established": False,
    }

    return metrics

def main():
    if len(sys.argv) == 5 and sys.argv[1] == "worker":
        name, direction = sys.argv[2], sys.argv[3]
        assert name in MODELS and direction in DIRECTIONS

        result = evaluate(name, direction)
        atomic_json(Path(sys.argv[4]), result)
        return

    assert len(sys.argv) == 1

    OUT.mkdir(parents=True, exist_ok=True)

    for name in MODELS:
        for direction in DIRECTIONS:
            label = f"{name}_{direction}_fp16"
            folder = OUT / label
            folder.mkdir(exist_ok=True)
            complete = folder / "result.json"

            if complete.exists():
                obj = json.loads(complete.read_text())
                assert obj["protocol"] == PROTOCOL
                assert obj["model"] == name
                assert obj["direction"] == direction
                assert obj["context_count"] == CONTEXT_COUNT
                print("SKIP VERIFIED:", label)
                continue

            attempt = 1
            while (folder / f"attempt_{attempt:03d}").exists():
                attempt += 1

            run = folder / f"attempt_{attempt:03d}"
            run.mkdir()

            result = run / "result.json"

            env = {
                **os.environ,
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }

            print("RUN:", label, flush=True)

            with (run / "stdout.txt").open("x") as out, \
                 (run / "stderr.txt").open("x") as err:
                try:
                    process = subprocess.run(
                        [
                            sys.executable, "-u",
                            str(Path(__file__).resolve()),
                            "worker", name, direction, str(result)
                        ],
                        env=env,
                        stdout=out,
                        stderr=err,
                        timeout=1800,
                    )
                    rc = process.returncode
                except subprocess.TimeoutExpired:
                    rc = 124

            if rc != 0 or not result.is_file():
                print("FAILED:", label, "exit:", rc)
                print(
                    (run / "stderr.txt")
                    .read_text(errors="replace")[-2000:]
                )
                raise SystemExit(1)

            obj = json.loads(result.read_text())
            assert obj["protocol"] == PROTOCOL
            assert obj["model"] == name
            assert obj["direction"] == direction
            assert obj["context_count"] == CONTEXT_COUNT

            os.link(result, complete)

            print(
                "COMMITTED:", label,
                "VALID_POSITIONS:",
                obj["valid_next_token_positions"],
                "TOKEN_AGREEMENT:",
                obj["token_agreement"],
                "CE_DELTA:",
                obj["cross_entropy_delta"],
                "KL:",
                obj["kl_fp32_to_fp16_nats"],
                "PPL_FP32:",
                obj["reference_perplexity"],
                "PPL_FP16:",
                obj["fp16_perplexity"],
                flush=True,
            )

    print("\nG4.13 TEXT QUALITY: PASS")
    print("OUTPUT:", OUT)
    print("OLD DATASETS MODIFIED: NO")
    print("POWER BENCHMARK RERUN: NO")

if __name__ == "__main__":
    main()
