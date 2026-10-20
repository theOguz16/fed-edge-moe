import csv
import hashlib
import itertools
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()
BASE = Path("results/benchmark_v3").resolve()

SRC = BASE / (
    "energy_repeats/"
    "t20_8_20261017T193026Z_2438"
)
DB = BASE / (
    "dataset_extensions/"
    "t19_3_20261017T183826Z_2438/"
    "dataset_v3_extended.sqlite"
)

NAMES = ("full_cpu", "full_mps", "mps_cpu_cut5")
BITS = {
    "full_cpu": "CCCCCC",
    "full_mps": "AAAAAA",
    "mps_cpu_cut5": "AAAAAC",
}
COLORS = {
    "full_cpu": "#c46a47",
    "full_mps": "#3479bd",
    "mps_cpu_cut5": "#25846a",
}
LABELS = {
    "full_cpu": "Full CPU",
    "full_mps": "Full MPS",
    "mps_cpu_cut5": "MPS to CPU cut 5",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=2, ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def write_csv(path, rows):
    assert rows
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def power_frames(path):
    regex = re.compile(
        r"^\s*(CPU Power|GPU Power|ANE Power|"
        r"Combined Power \(CPU \+ GPU \+ ANE\))"
        r":\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b"
    )
    names = {
        "CPU Power": "cpu",
        "GPU Power": "gpu",
        "ANE Power": "ane",
        "Combined Power (CPU + GPU + ANE)":
            "combined",
    }
    frames = []
    current = {}

    for line in path.read_text(
        errors="replace"
    ).splitlines():
        match = regex.match(line)
        if not match:
            continue

        label, number, unit = match.groups()
        value = float(number)
        if unit == "mW":
            value /= 1000

        key = names[label]
        current[key] = value

        if key == "combined":
            assert all(
                k in current
                for k in ("cpu", "gpu", "ane", "combined")
            )
            frames.append(dict(current))
            current = {}

    return frames

def mean(frames, name):
    return statistics.mean(
        row[name] for row in frames
    )

assert SRC.is_dir()
assert DB.is_file()

source_checks = json.loads(
    (SRC / "checksums.json").read_text()
)

for relative, expected in source_checks.items():
    assert sha(SRC / relative) == expected, (
        f"Source checksum mismatch: {relative}"
    )

plan = json.loads(
    (SRC / "experiment_plan.json").read_text()
)

orders = plan["orders"]
assert orders == [
    list(x) for x in itertools.permutations(NAMES)
]

rows = []
records = {}

for round_number, order in enumerate(orders, 1):
    for execution_order, name in enumerate(order, 1):
        folder = (
            SRC / f"round_{round_number:02d}" / name
        )
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text())

        assert summary["round"] == round_number
        assert summary["mapping"] == name
        assert summary["numerical_pass"] is True

        for relative, expected in (
            summary["artifact_sha256"].items()
        ):
            assert sha(SRC / relative) == expected

        attempt = folder / summary["attempt"]

        worker_path = attempt / "worker_result.json"
        worker = json.loads(worker_path.read_text())

        assert worker["model"] == "resnet50"
        assert worker["mapping"] == name
        assert worker["numerical_pass"] is True
        assert worker["top1_agreement"] is True
        assert worker["total_completed"] == len(
            worker["completion_timestamps_ns"]
        )
        assert worker["total_completed"] == len(
            worker["inference_durations_ns"]
        )

        before = power_frames(
            attempt / "idle_before.txt"
        )[1:-1]
        active = power_frames(
            attempt / "active.txt"
        )[2:-2]
        after = power_frames(
            attempt / "idle_after.txt"
        )[1:-1]

        assert len(before) >= 2
        assert len(active) >= 8
        assert len(after) >= 2

        active_w = mean(active, "combined")
        idle_before = mean(before, "combined")
        idle_after = mean(after, "combined")
        idle_w = mean(
            before + after, "combined"
        )

        assert math.isclose(
            active_w,
            summary["combined_active_watts"],
            abs_tol=1e-9
        )
        assert math.isclose(
            idle_before,
            summary["idle_before_watts"],
            abs_tol=1e-9
        )
        assert math.isclose(
            idle_after,
            summary["idle_after_watts"],
            abs_tol=1e-9
        )

        throughput = (
            summary["completed_images"]
            / summary["window_seconds"]
        )
        gross = active_w / throughput
        adjusted = (
            active_w - idle_w
        ) / throughput

        assert math.isclose(
            throughput,
            summary["throughput_images_per_second"],
            abs_tol=1e-9
        )
        assert math.isclose(
            gross,
            summary["gross_joules_per_image"],
            abs_tol=1e-9
        )
        assert math.isclose(
            adjusted,
            summary["idle_adjusted_joules_per_image"],
            abs_tol=1e-9
        )

        row = {
            "round": round_number,
            "execution_order": execution_order,
            "mapping": name,
            "cpu_threads": 4,
            "batch_size": 1,
            "precision": "fp32",
            "throughput_images_s": throughput,
            "cpu_power_w": mean(active, "cpu"),
            "gpu_power_w": mean(active, "gpu"),
            "combined_power_w": active_w,
            "gross_energy_j_image": gross,
            "idle_before_w": idle_before,
            "idle_after_w": idle_after,
            "idle_drift_w": (
                idle_after - idle_before
            ),
            "idle_adjusted_energy_j_image": adjusted,
            "power_samples_used": len(active),
            "measurement_status":
                "exploratory_repeated",
            "summary_sha256": sha(summary_path),
            "worker_sha256": sha(worker_path),
        }

        rows.append(row)
        records[(round_number, name)] = row

assert len(rows) == 18

print("=== T20.9 INDEPENDENT ENERGY AUDIT ===")
print("SOURCE ARTIFACT SHA256: PASS")
print("18/18 RUNS VERIFIED: PASS")
print("POWER FORMULAS RECOMPUTED: PASS")

comparisons = []

for round_number in range(1, 7):
    hybrid = records[
        (round_number, "mps_cpu_cut5")
    ]
    mps = records[
        (round_number, "full_mps")
    ]

    comparisons.append({
        "round": round_number,
        "energy_delta_percent": 100 * (
            hybrid["gross_energy_j_image"]
            / mps["gross_energy_j_image"]
            - 1
        ),
        "throughput_delta_percent": 100 * (
            hybrid["throughput_images_s"]
            / mps["throughput_images_s"]
            - 1
        ),
        "hybrid_energy_win": int(
            hybrid["gross_energy_j_image"]
            < mps["gross_energy_j_image"]
        ),
    })

wins = sum(
    r["hybrid_energy_win"]
    for r in comparisons
)
assert wins == 2

paired_median = statistics.median(
    r["energy_delta_percent"]
    for r in comparisons
)

write_csv(OUT / "energy_measurements.csv", rows)
write_csv(OUT / "paired_comparisons.csv", comparisons)

print("HYBRID ENERGY WINS:", f"{wins}/6")
print(
    "PAIRED MEDIAN ENERGY DELTA:",
    f"{paired_median:+.3f}%"
)

for name in NAMES:
    group = [
        r for r in rows
        if r["mapping"] == name
    ]
    values = [
        r["gross_energy_j_image"]
        for r in group
    ]
    print(
        f"{name}: "
        f"median={statistics.median(values):.5f} "
        f"range=[{min(values):.5f},"
        f"{max(values):.5f}] J/image"
    )

# Link these sustained measurement episodes to
# T17 configuration identities without changing
# the queued/completed status of any task.

source = sqlite3.connect(
    f"file:{DB}?mode=ro",
    uri=True
)
source.execute("PRAGMA foreign_keys=ON")

assert source.execute(
    "PRAGMA integrity_check"
).fetchone()[0] == "ok"

original_states = dict(source.execute(
    "SELECT state,COUNT(*) FROM tasks "
    "GROUP BY state"
).fetchall())

assert original_states == {
    "completed": 6,
    "planned": 327330,
}

task_ids = {}

for name, bits in BITS.items():
    matched = source.execute(
        """
        SELECT t.task_id
        FROM tasks t
        JOIN placements p
          ON t.placement_id=p.placement_id
        WHERE p.model='resnet50'
          AND p.group_mapping=?
          AND t.cpu_threads=4
          AND t.workload_id='batch1_micro1'
          AND t.execution_mode='sequential'
          AND t.hardware_id='mac_m4'
        """,
        (bits,)
    ).fetchall()

    assert len(matched) == 1
    task_ids[name] = matched[0][0]

new_db = OUT / "dataset_v3_with_energy.sqlite"
temporary = OUT / "dataset_v3_with_energy.sqlite.tmp"

destination = sqlite3.connect(str(temporary))

try:
    source.backup(destination)
    assert destination.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"
finally:
    destination.close()
    source.close()

db = sqlite3.connect(str(temporary))
db.execute("PRAGMA foreign_keys=ON")
db.execute("PRAGMA synchronous=FULL")

db.execute("""
CREATE TABLE energy_measurements (
    measurement_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    source_campaign TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    execution_order INTEGER NOT NULL,
    mapping TEXT NOT NULL,
    throughput_images_s REAL NOT NULL,
    cpu_power_w REAL NOT NULL,
    gpu_power_w REAL NOT NULL,
    combined_power_w REAL NOT NULL,
    gross_energy_j_image REAL NOT NULL,
    idle_before_w REAL NOT NULL,
    idle_after_w REAL NOT NULL,
    idle_adjusted_energy_j_image REAL NOT NULL,
    measurement_status TEXT NOT NULL,
    power_boundary TEXT NOT NULL,
    summary_path TEXT NOT NULL,
    summary_sha256 TEXT NOT NULL,
    worker_sha256 TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id),
    UNIQUE(source_campaign,round_number,mapping)
)
""")

db.execute("BEGIN IMMEDIATE")
try:
    for row in rows:
        identity = (
            "t20_8|"
            + str(row["round"])
            + "|"
            + row["mapping"]
        )
        mid = "energy_" + hashlib.sha256(
            identity.encode()
        ).hexdigest()

        summary_path = (
            SRC
            / f"round_{row['round']:02d}"
            / row["mapping"]
            / "summary.json"
        )

        db.execute(
            """
            INSERT INTO energy_measurements
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                mid,
                task_ids[row["mapping"]],
                "T20_8",
                row["round"],
                row["execution_order"],
                row["mapping"],
                row["throughput_images_s"],
                row["cpu_power_w"],
                row["gpu_power_w"],
                row["combined_power_w"],
                row["gross_energy_j_image"],
                row["idle_before_w"],
                row["idle_after_w"],
                row["idle_adjusted_energy_j_image"],
                row["measurement_status"],
                "Apple M4 powermetrics CPU+GPU+ANE",
                str(summary_path),
                row["summary_sha256"],
                row["worker_sha256"],
            )
        )
    db.commit()
except BaseException:
    db.rollback()
    raise

assert db.execute(
    "SELECT COUNT(*) FROM energy_measurements"
).fetchone()[0] == 18

assert dict(db.execute(
    "SELECT state,COUNT(*) FROM tasks "
    "GROUP BY state"
).fetchall()) == original_states

assert db.execute(
    "PRAGMA foreign_key_check"
).fetchall() == []

assert db.execute(
    "PRAGMA integrity_check"
).fetchone()[0] == "ok"

db.close()
os.replace(temporary, new_db)

# Publication-oriented, dependency-free scatter plot.
W, H = 1000, 610
left, right = 115, 930
top, bottom = 110, 510

def xy(t, e):
    x = left + (t - 65) / 20 * (right - left)
    y = bottom - (e - 0.07) / 0.08 * (bottom - top)
    return x, y

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<text x="48" y="43" font-family="Arial" '
    'font-size="24" font-weight="bold">'
    'ResNet50 — throughput vs. gross energy</text>',
    '<text x="48" y="70" font-family="Arial" '
    'font-size="14" fill="#555">'
    'Apple M4 | n=6 independent processes per mapping '
    '| exploratory, approximately aligned power sampling</text>',
]

for t in (65, 70, 75, 80, 85):
    x, _ = xy(t, 0.07)
    svg.extend([
        f'<line x1="{x:.1f}" y1="{top}" '
        f'x2="{x:.1f}" y2="{bottom}" '
        'stroke="#ececec"/>',
        f'<text x="{x:.1f}" y="533" '
        'text-anchor="middle" font-size="13" '
        f'font-family="Arial">{t}</text>',
    ])

for e in (0.07, 0.09, 0.11, 0.13, 0.15):
    _, y = xy(65, e)
    svg.extend([
        f'<line x1="{left}" y1="{y:.1f}" '
        f'x2="{right}" y2="{y:.1f}" '
        'stroke="#ececec"/>',
        f'<text x="100" y="{y+4:.1f}" '
        'text-anchor="end" font-size="13" '
        f'font-family="Arial">{e:.2f}</text>',
    ])

svg.extend([
    f'<line x1="{left}" y1="{top}" '
    f'x2="{left}" y2="{bottom}" stroke="#333"/>',
    f'<line x1="{left}" y1="{bottom}" '
    f'x2="{right}" y2="{bottom}" stroke="#333"/>',
    '<text x="522" y="566" text-anchor="middle" '
    'font-family="Arial" font-size="17">'
    'Throughput (images/second) — higher is better</text>',
    '<text x="28" y="315" transform="rotate(-90 28 315)" '
    'text-anchor="middle" font-family="Arial" font-size="17">'
    'Gross energy (J/image) — lower is better</text>',
])

for name in NAMES:
    color = COLORS[name]
    for row in rows:
        if row["mapping"] != name:
            continue
        x, y = xy(
            row["throughput_images_s"],
            row["gross_energy_j_image"]
        )
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" '
            f'fill="{color}" fill-opacity="0.82" '
            f'stroke="white" stroke-width="1.5">'
            f'<title>{LABELS[name]}, round {row["round"]}</title>'
            '</circle>'
        )

for index, name in enumerate(NAMES):
    x = 145 + index * 280
    svg.extend([
        f'<circle cx="{x}" cy="588" r="6" '
        f'fill="{COLORS[name]}"/>',
        f'<text x="{x+14}" y="593" '
        f'font-family="Arial" font-size="14">'
        f'{LABELS[name]}</text>',
    ])

svg.append("</svg>")

figure = OUT / "throughput_energy_scatter.svg"
figure.write_text(
    "\n".join(svg), encoding="utf-8"
)

# Package the actual raw observations rather
# than keeping database pointers alone.
archive = OUT / "t20_8_raw_artifacts.zip"

with zipfile.ZipFile(
    archive, "w",
    compression=zipfile.ZIP_DEFLATED
) as z:
    for relative in sorted(source_checks):
        z.write(
            SRC / relative,
            arcname="t20_8/" + relative
        )
    z.write(
        SRC / "checksums.json",
        arcname="t20_8/checksums.json"
    )

with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == (
        len(source_checks) + 1
    )

report = {
    "protocol": "T20_9_energy_import_and_graph",
    "source_campaign": str(SRC),
    "source_database": str(DB),
    "new_database": str(new_db),
    "measurement_rows_imported": 18,
    "original_tasks_preserved": 327336,
    "original_completed_tasks_preserved": 6,
    "hybrid_energy_wins_vs_full_mps": wins,
    "paired_median_energy_delta_percent":
        paired_median,
    "power_boundary":
        "Apple M4 CPU+GPU+ANE estimated",
    "time_alignment": "approximate",
    "statistical_status": "exploratory",
    "independent_runs_per_mapping": 6,
    "old_database_modified": False,
    "new_inference_runs": 0,
    "raw_archive": archive.name,
    "figure": figure.name,
    "figures_are_not_significance_tests": True,
}

write_json(OUT / "analysis_summary.json", report)

files = [
    p for p in OUT.iterdir()
    if p.is_file()
    and p.name != "checksums.json"
]

write_json(
    OUT / "checksums.json",
    {p.name: sha(p) for p in files}
)

for name, digest in json.loads(
    (OUT / "checksums.json").read_text()
).items():
    assert sha(OUT / name) == digest

print("\n=== T20.9 DATASET INTEGRATION ===")
print("ENERGY MEASUREMENT ROWS: 18")
print("ORIGINAL TASKS PRESERVED: 327336")
print("ORIGINAL COMPLETED TASKS PRESERVED: 6")
print("SQLITE INTEGRITY: PASS")
print("RAW ARTIFACT ZIP: PASS")
print("SCIENTIFIC SVG GRAPH: PASS")
print("CHECKSUMS: PASS")
print("NEW INFERENCE MEASUREMENTS: 0")
print("OLD DATABASE MODIFIED: NO")
print("T20.9 ENERGY DATASET INTEGRATION: PASS")
print("RESULT DIRECTORY:", OUT)
