import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path("results/benchmark_v3").resolve()
DB = BASE / "production_campaigns/t23_mac_v1/dataset_v3_t23_mac.sqlite"
OUT = BASE / "pipeline_dataset_exports" / datetime.now(
    timezone.utc
).strftime("t26_merged_%Y%m%dT%H%M%S%fZ")

assert DB.is_file()
OUT.mkdir(parents=True, exist_ok=False)

modes = []
pairs = []

def norm_direction(value):
    return {
        "cpu_to_mps": "cpu_mps",
        "mps_to_cpu": "mps_cpu",
        "cpu_mps": "cpu_mps",
        "mps_cpu": "mps_cpu",
    }.get(value, str(value))

def add_case(info, results):
    assert len(results) == 2, (info, len(results))
    selected = {r["mode"]: r for r in results}
    assert set(selected) == {"sequential", "pipeline"}, info

    unit = "images" if info["family"] == "vision" else "tokens"
    energy_unit = "J/image" if info["family"] == "vision" else "J/token"

    for mode in ("sequential", "pipeline"):
        r = selected[mode]
        throughput = float(r["throughput"])
        energy = float(r["energy"])
        assert throughput > 0 and energy > 0

        modes.append({
            **info,
            "mode": mode,
            "throughput_per_second": throughput,
            "throughput_unit": unit + "/s",
            "energy_per_item": energy,
            "energy_unit": energy_unit,
            "cpu_power_w": float(r["cpu_power_w"]),
            "gpu_power_w": float(r["gpu_power_w"]),
            "median_wall_ms": float(r["median_wall_ms"]),
            "numerical_status": r["numerical_status"],
            "result_path": r["result_path"],
            "result_sha256": r["result_sha256"],
            "power_sha256": r["power_sha256"],
        })

    seq = selected["sequential"]
    pipe = selected["pipeline"]
    speedup = float(pipe["throughput"]) / float(seq["throughput"])
    energy_ratio = float(pipe["energy"]) / float(seq["energy"])

    assert speedup > 0 and energy_ratio > 0

    pairs.append({
        **info,
        "speedup_pipeline_over_sequential": speedup,
        "energy_pipeline_over_sequential": energy_ratio,
        "pipeline_faster": speedup > 1,
        "pipeline_energy_lower": energy_ratio < 1,
        "throughput_unit": unit + "/s",
        "energy_unit": energy_unit,
    })

def info(
    campaign, cid, model, family, direction,
    cut, mapping_bits, micro, qos, repeat,
    precision="fp32", batch=None, context=None,
):
    return {
        "campaign": campaign,
        "case_id": cid,
        "model": model,
        "family": family,
        "direction": norm_direction(direction),
        "cut": str(cut) if cut is not None else "",
        "mapping_bits": str(mapping_bits) if mapping_bits is not None else "",
        "microbatch": int(micro),
        "qos": str(qos),
        "repeat": int(repeat),
        "precision": precision,
        "batch_size": batch,
        "context_tokens": context,
        "power_scope": "estimated_cpu_gpu_soc_not_wall",
        "gpu_kernel_overlap_proven": False,
    }

with sqlite3.connect(
    f"file:{DB}?mode=ro", uri=True
) as db:
    db.row_factory = sqlite3.Row
    assert db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    a_cases = list(db.execute(
        "SELECT * FROM t26a_cases ORDER BY ordinal"
    ))
    b_cases = list(db.execute(
        "SELECT * FROM t26b_cases ORDER BY ordinal"
    ))
    c_cases = list(db.execute(
        "SELECT * FROM t26c_v2_cases ORDER BY ordinal"
    ))

    assert len(a_cases) == 48
    assert len(b_cases) == 192
    assert len(c_cases) == 288

    def fetch(table, cid):
        return list(db.execute(
            f"SELECT * FROM {table} WHERE case_id=?",
            (cid,)
        ))

    for c in a_cases:
        assert c["status"] == "completed", dict(c)
        rs = fetch("t26a_results", c["case_id"])
        add_case(
            info(
                "t26a", c["case_id"], "resnet50",
                "vision", c["mapping"], None,
                c["mapping_bits"], c["microbatch"],
                c["qos"], c["repeat"],
            ),
            [{
                "mode": r["mode"],
                "throughput": r["sustained_images_s"],
                "energy": r["gross_energy_j_image"],
                "cpu_power_w": r["cpu_power_w"],
                "gpu_power_w": r["gpu_power_w"],
                "median_wall_ms": r["batch_wall_median_ms"],
                "numerical_status":
                    "validated_in_prior_t26a_audit_not_stored_in_row",
                "result_path": r["result_path"],
                "result_sha256": r["result_sha256"],
                "power_sha256": r["power_raw_sha256"],
            } for r in rs],
        )

    for c in b_cases:
        assert c["status"] == "completed", dict(c)
        rs = fetch("t26b_results", c["case_id"])
        add_case(
            info(
                "t26b", c["case_id"], c["model"],
                "vision", c["direction"], c["cut"],
                c["mapping"], c["microbatch"],
                c["qos"], c["repeat"],
            ),
            [{
                "mode": r["mode"],
                "throughput": r["throughput_images_s"],
                "energy": r["gross_energy_j_image"],
                "cpu_power_w": r["cpu_power_w"],
                "gpu_power_w": r["gpu_power_w"],
                "median_wall_ms": r["batch_wall_median_ms"],
                "numerical_status": r["numerical_status"],
                "result_path": r["result_path"],
                "result_sha256": r["result_sha256"],
                "power_sha256": r["power_sha256"],
            } for r in rs],
        )

    for c in c_cases:
        if c["status"] == "completed":
            rs = fetch("t26c_v2_results", c["case_id"])
            campaign = "t26c_v2"
        else:
            assert c["status"] == "legacy_linked", dict(c)
            assert c["legacy_case_id"]
            rs = fetch("t26c_results", c["legacy_case_id"])
            campaign = "t26c_legacy"

        add_case(
            info(
                campaign, c["case_id"], c["model"],
                "text", c["direction"], c["cut"],
                None, c["microbatch_size"],
                c["qos"], c["repeat"],
                c["precision"], c["batch_size"],
                c["context_tokens"],
            ),
            [{
                "mode": r["mode"],
                "throughput": (
                    r["throughput_tokens_s"]
                ),
                "energy": (
                    r["energy_j_token"]
                ),
                "cpu_power_w": r["cpu_power_w"],
                "gpu_power_w": r["gpu_power_w"],
                "median_wall_ms": r["median_wall_ms"],
                "numerical_status": r["numerical_status"],
                "result_path": r["result_path"],
                "result_sha256": r["result_sha256"],
                "power_sha256": r["power_sha256"],
            } for r in rs],
        )

assert len(pairs) == 528, len(pairs)
assert len(modes) == 1056, len(modes)
assert len({r["case_id"] for r in pairs}) == 528
assert len({
    (r["case_id"], r["mode"]) for r in modes
}) == 1056

def save_csv(filename, rows):
    with (OUT / filename).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

save_csv("pipeline_modes_1056.csv", modes)
save_csv("pipeline_pairs_528.csv", pairs)

pq.write_table(
    pa.Table.from_pylist(modes),
    OUT / "pipeline_modes_1056.parquet",
    compression="zstd",
)
pq.write_table(
    pa.Table.from_pylist(pairs),
    OUT / "pipeline_pairs_528.parquet",
    compression="zstd",
)

grouped = defaultdict(list)
for row in pairs:
    grouped[row["model"]].append(row)

summary = []
for model, values in sorted(grouped.items()):
    speedups = [
        x["speedup_pipeline_over_sequential"]
        for x in values
    ]
    energies = [
        x["energy_pipeline_over_sequential"]
        for x in values
    ]
    summary.append({
        "model": model,
        "family": values[0]["family"],
        "paired_cases": len(values),
        "median_speedup": statistics.median(speedups),
        "median_energy_ratio": statistics.median(energies),
        "faster_cases": sum(x > 1 for x in speedups),
        "energy_saving_cases": sum(x < 1 for x in energies),
    })

save_csv("model_summary.csv", summary)

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_db": str(DB),
    "source_db_access": "read_only",
    "pair_count": len(pairs),
    "mode_count": len(modes),
    "legacy_text_pairs_included": 2,
    "scope": "Mac M4 FP32 hybrid inference pipeline",
    "throughput_units": {
        "vision": "images/s",
        "text": "prefill tokens/s",
    },
    "energy_scope": "estimated CPU+GPU SoC, not wall",
    "limitations": [
        "PyTorch native worker QoS not independently verified",
        "GPU kernel overlap not proven",
        "Cross-family absolute throughput is not comparable",
        "Energy data can be sensitive to power sampling and QoS",
        "T26A numerical status is not stored per result row",
        "Legacy DistilGPT2 measurements use earlier protocol",
    ],
    "models": summary,
}
(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [x["model"] for x in summary]
    xs = list(range(len(names)))

    for key, filename, ylabel in (
        ("median_speedup", "median_speedup.png",
         "Pipeline / sequential throughput"),
        ("median_energy_ratio", "median_energy_ratio.png",
         "Pipeline / sequential energy per item"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        values = [x[key] for x in summary]
        ax.bar(xs, values, color="#407ca4")
        ax.axhline(1, color="#bd5448",
                   linestyle="--", linewidth=1.5)
        ax.set_xticks(xs, names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("FedEdgeMoE Dataset V3 — Mac M4")
        for x, y in zip(xs, values):
            ax.annotate(
                f"{y:.3f}×", (x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)

    print("PLOTS: PASS")
except ImportError:
    print("PLOTS: SKIPPED — matplotlib not installed")

print("\nDATASET EXPORT: PASS")
print("PAIRED CASES:", len(pairs))
print("MODE MEASUREMENTS:", len(modes))
for item in summary:
    print(
        item["model"],
        "n=", item["paired_cases"],
        "speedup=", f"{item['median_speedup']:.3f}x",
        "energy=", f"{item['median_energy_ratio']:.3f}x",
    )
print("OUTPUT:", OUT)
print("SQLITE MODIFIED: NO")
print("BENCHMARK RERUN: NO")
