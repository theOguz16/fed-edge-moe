import csv
import hashlib
import statistics
from pathlib import Path

ROOT = Path("results")
OUT = ROOT / "qwen3_hard_q4_cuda_recovered_latency.csv"

SOURCES = {
    "light": (
        "qwen3_hard_q4_cuda_sweep.txt",
        128, 64, 1, 1
    ),
    "medium": (
        "qwen3_hard_q4_cuda_medium_sustained.txt",
        512, 128, 2, 42
    ),
    "heavy": (
        "qwen3_hard_q4_cuda_heavy_sustained.txt",
        1024, 128, 4, 21
    ),
}

rows_out = []

for profile, (filename, pp, tg, batch, expected_n) in SOURCES.items():
    path = ROOT / filename
    raw = path.read_bytes()

    assert raw.startswith((b"\xff\xfe", b"\xfe\xff")), (
        f"Expected UTF-16 with BOM: {filename}"
    )

    text = raw.decode("utf-16")
    assert "n_gpu_layers = 99" in text, filename

    latencies = []

    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue

        fields = [
            x.strip()
            for x in line.strip().strip("|").split("|")
        ]

        if len(fields) != 10:
            continue

        try:
            p, t, b, n_kv = map(int, fields[:4])
            t_pp = float(fields[4])
            t_tg = float(fields[6])
            t_total = float(fields[8])
        except ValueError:
            continue

        if (p, t, b) != (pp, tg, batch):
            continue

        assert n_kv == (pp + tg) * batch
        assert abs((t_pp + t_tg) - t_total) <= 0.005
        assert t_total > 0

        latencies.append(t_total)

    assert len(latencies) == expected_n, (
        f"{profile}: expected {expected_n}, "
        f"found {len(latencies)}"
    )

    record = {
        "profile": profile,
        "model": "Qwen3-1.7B",
        "device": "RTX 3050 Laptop",
        "backend": "llama.cpp/cuda",
        "precision": "q4_k_m",
        "quantization": "Q4_K_M",
        "context_tokens": pp,
        "output_tokens": tg,
        "batch": batch,
        "latency_sec": statistics.median(latencies),
        "latency_min_sec": min(latencies),
        "latency_max_sec": max(latencies),
        "observations": len(latencies),
        "latency_semantics": "llamacpp_batched_benchmark_total",
        "evidence_scope": (
            "single_sweep_observation"
            if profile == "light"
            else "sequential_sustained_observations"
        ),
        "source_file": filename,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }

    rows_out.append(record)

assert len(rows_out) == 3
assert [r["latency_sec"] for r in rows_out] == [
    0.588, 1.471, 2.968
]

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows_out[0].keys(),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows_out)

print("=== RECOVERED QWEN3 CUDA LATENCY ===")

for row in rows_out:
    print(
        f"{row['profile']:6} | "
        f"n={row['observations']:2} | "
        f"median={row['latency_sec']:.3f} s | "
        f"range=[{row['latency_min_sec']:.3f}, "
        f"{row['latency_max_sec']:.3f}] s"
    )

print("QWEN3 CUDA LATENCY RECOVERY: PASS")
print("Saved:", OUT)
