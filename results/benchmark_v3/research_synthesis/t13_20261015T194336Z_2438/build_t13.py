import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()

SOURCES = {
    "T05_numerical_validation":
        "results/benchmark_v3/validation/t05_20261015T173344Z_2438",
    "T06_structural_inventory":
        "results/benchmark_v3/inventory/inv_20261015T175122Z_2438",
    "T06_resnet50_latency":
        "results/benchmark_v3/latency_pilot/resnet50_20261015T175850Z_2438",
    "T06_convnext_latency":
        "results/benchmark_v3/latency_pilot/convnext_20261015T180355Z_2438",
    "T07_resnet50_hybrid":
        "results/benchmark_v3/hybrid_pilot/resnet50_20261015T181030Z_2438",
    "T07_convnext_hybrid":
        "results/benchmark_v3/hybrid_pilot/convnext_20261015T181523Z_2438",
    "T08_independent_repeats":
        "results/benchmark_v3/repeated_hybrid/paired_20261015T184048Z_79565_fix01",
    "T09_image_variability":
        "results/benchmark_v3/image_paired/paired16_20261015T185509Z_2438",
    "T10_thread_pilot":
        "results/benchmark_v3/thread_partition_pilot/resnet50_20261015T190022Z_2438",
    "T10_independent_repeats":
        "results/benchmark_v3/thread_partition_repeats/resnet50_20261015T190735Z_2438",
    "T11_convnext_threads":
        "results/benchmark_v3/thread_partition_convnext/t11_20261015T191957Z_2438",
    "T12_holdout_confirmation":
        "results/benchmark_v3/holdout_thread_confirmation/t12_20261015T192923Z_2438",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def loadcsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def savecsv(path, columns, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def close(a, b, tolerance=1e-8):
    return math.isclose(
        float(a), float(b),
        rel_tol=tolerance,
        abs_tol=tolerance
    )

def source(key):
    return Path(SOURCES[key])

def median(values):
    return statistics.median(values)

print("=== T13 SOURCE AUDIT ===", flush=True)

assert len(SOURCES) == 12

source_inventory = []
checked_paths = set()
verified_checksums = 0

for campaign, directory_text in SOURCES.items():
    directory = Path(directory_text)

    assert directory.is_dir(), (
        f"Missing source directory: {directory}"
    )

    checksum_files = sorted(
        directory.rglob("checksums.json")
    )

    assert checksum_files, (
        f"Missing checksum records: {directory}"
    )

    for checksum_file in checksum_files:
        checks = json.loads(
            checksum_file.read_text()
        )

        assert isinstance(checks, dict)
        assert checks

        for name, expected in checks.items():
            relative = Path(name)

            assert not relative.is_absolute()
            assert ".." not in relative.parts

            target = checksum_file.parent / relative

            assert target.is_file(), (
                f"Missing checked file: {target}"
            )
            assert sha(target) == expected, (
                f"SHA256 mismatch: {target}"
            )

            checked_paths.add(str(target.resolve()))
            verified_checksums += 1

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file()
    )

    for path in files:
        row_count = ""

        if path.suffix.lower() == ".csv":
            with path.open(newline="") as f:
                rows = csv.reader(f)
                next(rows, None)
                row_count = sum(1 for _ in rows)

        source_inventory.append({
            "campaign": campaign,
            "relative_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "listed_in_source_checksums": int(
                str(path.resolve()) in checked_paths
            ),
            "csv_data_rows": row_count,
        })

    print(
        f"{campaign}: PASS "
        f"({len(files)} files, "
        f"{len(checksum_files)} checksum manifests)",
        flush=True
    )

print(
    f"SOURCE CAMPAIGNS: {len(SOURCES)}/12 PASS",
    flush=True
)
print(
    f"VERIFIED SHA256 ENTRIES: {verified_checksums}",
    flush=True
)

# The source checksum manifests are preserved. T13 also
# independently hashes every indexed file, including files
# that were not listed in an original checksum manifest.

print("\n=== RECOMPUTING SCIENTIFIC FINDINGS ===")

FINDING_FIELDS = [
    "stage", "model", "cpu_threads", "mapping",
    "comparison_unit", "paired_comparisons",
    "hybrid_faster_count",
    "median_pair_delta_percent",
    "median_process_delta_percent",
    "minimum_pair_delta_percent",
    "maximum_pair_delta_percent",
    "independent_processes",
]

findings = []

def validate_pairs(rows):
    assert rows

    for row in rows:
        baseline = float(row["full_mps_ms"])
        hybrid = float(row["hybrid_ms"])

        assert baseline > 0
        assert hybrid > 0

        expected_delta = 100 * (
            hybrid / baseline - 1
        )
        expected_win = int(hybrid < baseline)

        assert close(
            row["delta_percent"],
            expected_delta
        )
        assert int(row["hybrid_faster"]) == expected_win

def add_finding(
    stage, model, threads, mapping,
    rows, comparison_unit, processes,
    process_median=None
):
    validate_pairs(rows)

    deltas = [
        float(row["delta_percent"])
        for row in rows
    ]

    if process_median is None:
        process_median = median(deltas)

    finding = {
        "stage": stage,
        "model": model,
        "cpu_threads": threads,
        "mapping": mapping,
        "comparison_unit": comparison_unit,
        "paired_comparisons": len(rows),
        "hybrid_faster_count": sum(
            int(row["hybrid_faster"])
            for row in rows
        ),
        "median_pair_delta_percent": median(deltas),
        "median_process_delta_percent": process_median,
        "minimum_pair_delta_percent": min(deltas),
        "maximum_pair_delta_percent": max(deltas),
        "independent_processes": processes,
    }

    findings.append(finding)
    return finding

# T08: one image, eight independent processes.
t08 = loadcsv(
    source("T08_independent_repeats")
    / "paired_comparison.csv"
)
assert len(t08) == 16

t08_results = {}

for model in ("convnext_base", "convnext_large"):
    selected = [
        r for r in t08
        if r["model"] == model
    ]
    assert len(selected) == 8
    assert {
        int(r["process_repeat"]) for r in selected
    } == set(range(1, 9))

    t08_results[model] = add_finding(
        "T08", model, 4, "selected_hybrid",
        selected, "independent_process", 8
    )

# T09: sixteen images, two independent processes.
t09 = loadcsv(
    source("T09_image_variability")
    / "paired_images.csv"
)
assert len(t09) == 64

t09_results = {}

for model in ("convnext_base", "convnext_large"):
    selected = [
        r for r in t09
        if r["model"] == model
    ]
    assert len(selected) == 32
    assert len({
        (int(r["process_repeat"]), int(r["image_index"]))
        for r in selected
    }) == 32

    t09_results[model] = add_finding(
        "T09", model, 4, "selected_hybrid",
        selected, "image_process_pair", 2
    )

# T10: independent ResNet50 thread/partition study.
t10 = loadcsv(
    source("T10_independent_repeats")
    / "paired_comparison.csv"
)
assert len(t10) == 32

t10_results = {}

for threads in (2, 4, 6, 10):
    for mapping in ("CPU_TO_MPS", "MPS_TO_CPU"):
        selected = [
            r for r in t10
            if int(r["thread_count"]) == threads
            and r["hybrid_mapping"] == mapping
        ]

        assert len(selected) == 4
        assert {
            int(r["round_number"]) for r in selected
        } == {1, 2, 3, 4}

        key = (threads, mapping)

        t10_results[key] = add_finding(
            "T10.3", "resnet50", threads, mapping,
            selected, "independent_process", 4
        )

# T11: ConvNeXt thread-by-placement comparisons.
t11 = loadcsv(
    source("T11_convnext_threads")
    / "paired_comparison.csv"
)
assert len(t11) == 72

t11_results = {}

for model in ("convnext_base", "convnext_large"):
    for threads in (1, 2, 4, 6, 8, 10):
        for mapping in ("CPU_TO_MPS", "MPS_TO_CPU"):
            selected = [
                r for r in t11
                if r["model"] == model
                and int(r["threads"]) == threads
                and r["mapping"] == mapping
            ]

            assert len(selected) == 3
            assert {
                int(r["round"]) for r in selected
            } == {1, 2, 3}

            key = (model, threads, mapping)

            t11_results[key] = add_finding(
                "T11", model, threads, mapping,
                selected, "independent_process", 3
            )

# T12: independent holdout image experiment.
t12 = loadcsv(
    source("T12_holdout_confirmation")
    / "paired_images.csv"
)
assert len(t12) == 128

t12_results = {}
t12_rounds = {}
t12_images = {}

for threads in (1, 4):
    selected = [
        r for r in t12
        if int(r["threads"]) == threads
    ]

    assert len(selected) == 64
    assert len({
        (int(r["round"]), int(r["image_index"]))
        for r in selected
    }) == 64

    round_medians = []

    for round_no in (1, 2, 3, 4):
        values = [
            float(r["delta_percent"])
            for r in selected
            if int(r["round"]) == round_no
        ]

        assert len(values) == 16
        round_medians.append(median(values))

    favorable_images = 0

    for image_index in range(16):
        image_rows = [
            r for r in selected
            if int(r["image_index"]) == image_index
        ]
        assert len(image_rows) == 4

        assert sorted(
            int(r["full_mps_order"])
            for r in image_rows
        ) == [1, 1, 2, 2]

        image_median = median(
            float(r["delta_percent"])
            for r in image_rows
        )
        favorable_images += image_median < 0

    t12_rounds[threads] = round_medians
    t12_images[threads] = favorable_images

    t12_results[threads] = add_finding(
        "T12", "convnext_large", threads,
        "MPS_TO_CPU",
        selected, "image_process_pair", 4,
        process_median=median(round_medians)
    )

# Sanity checks against previously verified findings.
assert t08_results["convnext_base"][
    "hybrid_faster_count"
] == 0
assert t08_results["convnext_large"][
    "hybrid_faster_count"
] == 4

assert t09_results["convnext_base"][
    "hybrid_faster_count"
] == 4
assert t09_results["convnext_large"][
    "hybrid_faster_count"
] == 14

assert all(
    t10_results[(t, "CPU_TO_MPS")][
        "hybrid_faster_count"
    ] == 0
    for t in (2, 4, 6, 10)
)

candidate = t11_results[
    ("convnext_large", 1, "MPS_TO_CPU")
]
assert candidate["hybrid_faster_count"] == 3
assert abs(
    candidate["median_pair_delta_percent"] + 1.654
) < 0.02

assert t12_results[1]["hybrid_faster_count"] == 21
assert t12_results[4]["hybrid_faster_count"] == 24
assert t12_images == {1: 3, 4: 5}
assert all(
    d > 0
    for values in t12_rounds.values()
    for d in values
)

assert len(findings) == 38

# The T10 pilot outlier can be evaluated directly
# against the independently repeated runs.
t10_pilot = loadcsv(
    source("T10_thread_pilot")
    / "latency_summary.csv"
)
t10_repeats = loadcsv(
    source("T10_independent_repeats")
    / "all_summary.csv"
)

pilot_cpu4 = [
    float(r["median_ms"])
    for r in t10_pilot
    if int(r["thread_count"]) == 4
    and r["mapping"] == "FULL_CPU"
]

repeat_cpu4 = sorted(
    float(r["median_ms"])
    for r in t10_repeats
    if int(r["thread_count"]) == 4
    and r["mapping"] == "FULL_CPU"
)

assert len(pilot_cpu4) == 1
assert len(repeat_cpu4) == 4
assert abs(pilot_cpu4[0] - 25.170) < 0.02
assert repeat_cpu4[-1] < 15

# Confirm holdout exclusion independently.
validation = json.loads(
    (
        source("T05_numerical_validation")
        / "metadata.json"
    ).read_text()
)
holdout = json.loads(
    (
        source("T12_holdout_confirmation")
        / "holdout_images.json"
    ).read_text()
)

prior_paths = {
    r["path"] for r in validation["images"]
}
prior_hashes = {
    r["sha256"] for r in validation["images"]
}

assert len(prior_paths) == 16
assert len(holdout) == 16
assert all(
    r["path"] not in prior_paths
    and r["sha256"] not in prior_hashes
    for r in holdout
)

assert len({
    r["sha256"] for r in holdout
}) == 16

print("T08–T12 PAIRED RESULTS: PASS")
print("T10 FULL CPU OUTLIER REVIEW: PASS")
print("T12 HOLDOUT EXCLUSION: PASS")
print("DERIVED FINDING ROWS: 38")

def fmt(value):
    return f"{value:+.3f}%"

def outcome(stage, model, threads=None, mapping=None):
    candidates = [
        r for r in findings
        if r["stage"] == stage
        and r["model"] == model
        and (
            threads is None
            or r["cpu_threads"] == threads
        )
        and (
            mapping is None
            or r["mapping"] == mapping
        )
    ]
    assert len(candidates) == 1
    return candidates[0]

def short(row, process_median=False):
    delta = (
        row["median_process_delta_percent"]
        if process_median
        else row["median_pair_delta_percent"]
    )
    return (
        f"{row['hybrid_faster_count']}/"
        f"{row['paired_comparisons']} hızlı, "
        f"medyan fark {fmt(delta)}"
    )

lines = [
    "# FedEdgeMoE Dataset V3 — T05–T12 Bilimsel Sentez",
    "",
    f"Üretim zamanı (UTC): "
    f"{datetime.now(timezone.utc).isoformat()}",
    "",
    "## 1. Kapsam",
    "",
    "Bu özet, Mac M4 üzerindeki ResNet50, "
    "ConvNeXt-Base ve ConvNeXt-Large deneylerini kapsar.",
    "FP32 ve batch=1 kullanılmıştır. Zaman ölçümleri "
    "sequential E2E inference içindir; CPU giriş/çıkışı "
    "ile gerekli CPU/MPS geçişleri dahildir. "
    "Preprocessing ve ağırlık yerleşimi hariçtir.",
    "",
    "T05 sayısal doğrulama, T06 yapısal envanter ve "
    "stage latency, T07 hibrit pilot, T08–T09 "
    "bağımsız tekrarlar ve görüntü çeşitliliği, "
    "T10–T11 thread × partition çalışmaları, "
    "T12 yeni görüntülerle holdout doğrulamasıdır.",
    "",
    "## 2. Doğrulanmış kritik sonuçlar",
    "",
    "| Aşama | Model / koşul | Bulgular |",
    "|---|---|---|",
]

highlights = [
    (
        "T08", "ConvNeXt-Base, cut 1",
        outcome("T08", "convnext_base"),
        False
    ),
    (
        "T08", "ConvNeXt-Large, cut 5",
        outcome("T08", "convnext_large"),
        False
    ),
    (
        "T09", "ConvNeXt-Base, 16 görüntü",
        outcome("T09", "convnext_base"),
        False
    ),
    (
        "T09", "ConvNeXt-Large, 16 görüntü",
        outcome("T09", "convnext_large"),
        False
    ),
    (
        "T10.3", "ResNet50, 4 thread, cut 5",
        outcome(
            "T10.3", "resnet50", 4, "MPS_TO_CPU"
        ),
        False
    ),
    (
        "T11", "ConvNeXt-Large, 1 thread, cut 5",
        outcome(
            "T11", "convnext_large", 1, "MPS_TO_CPU"
        ),
        False
    ),
    (
        "T12", "ConvNeXt-Large, 1 thread, holdout",
        outcome(
            "T12", "convnext_large", 1, "MPS_TO_CPU"
        ),
        True
    ),
    (
        "T12", "ConvNeXt-Large, 4 thread, holdout",
        outcome(
            "T12", "convnext_large", 4, "MPS_TO_CPU"
        ),
        True
    ),
]

for stage, label, row, use_process in highlights:
    lines.append(
        f"| {stage} | {label} | "
        f"{short(row, use_process)} |"
    )

lines.extend([
    "",
    "Negatif latency farkı hibritin daha hızlı olduğunu, "
    "pozitif fark Full MPS'nin daha hızlı olduğunu gösterir.",
    "",
    "**T08:** Seçilmiş hibritlerin ilk pilotlarda görülen "
    "küçük hızlanmaları bağımsız süreç tekrarlarında "
    "tutarlı şekilde korunmamıştır.",
    "",
    "**T09:** Farklar farklı görüntü-süreç çiftlerinde "
    "değişmektedir. Sabit tensor şekilleri nedeniyle "
    "bu dalgalanma doğrudan görüntü içeriğine "
    "atfedilemez.",
    "",
    "**T10:** ResNet50'nin 4-thread Full CPU "
    f"pilot medyanı {pilot_cpu4[0]:.3f} ms idi. "
    "Bağımsız tekrarlar "
    + ", ".join(
        f"{v:.3f} ms" for v in repeat_cpu4
    )
    + " değerlerini verdi. Pilot anormalliği "
    "kararlı performans olarak doğrulanmadı.",
    "",
    "**T11:** ConvNeXt-Large, 1-thread, cut 5 "
    "yerleşimi üç keşif tekrarında hızlıydı. "
    "Bu sonuç, çok sayıda taranan konfigürasyon "
    "içinden seçilmiş keşif bulgusudur.",
    "",
    "**T12:** Aynı aday, önceden kullanılan "
    "16 görüntü dışlanarak seçilen 16 yeni görüntüde "
    "yeniden sınandı. "
    "1-thread ayarında süreç medyanlarının dördü de "
    "pozitiftir: "
    + ", ".join(fmt(v) for v in t12_rounds[1])
    + ". Dolayısıyla T11 avantajı holdout üzerinde "
    "doğrulanmamıştır.",
    "",
    "## 3. Analiz birimi ve sınırlamalar",
    "",
    "- 10–15 arası ardışık timed call bağımsız "
    "deney olarak sayılmamıştır.",
    "- T08 ve T11'de bağımsız süreç medyanları; "
    "T09'da görüntü-süreç çiftleri; T12'de ise "
    "görüntü çiftleri ve süreç başına medyanlar "
    "ayrı seviyeler olarak ele alınmıştır.",
    "- T12'nin 64 çift/ayar gözlemi, 64 bağımsız "
    "Python süreci veya 64 farklı görüntü değildir.",
    "- Cihaz, sistem yükü ve termal durum etkileri "
    "tam kontrol edilmemiştir.",
    "- Bu çalışmalar enerji, peak memory, throughput "
    "veya overlap/pipeline hızlanmasını doğrulamaz.",
    "- İncelenen tek-kesimli CPU/MPS yerleşimlerinden "
    "hareketle tüm olası scheduler politikaları "
    "hakkında genel üstünlük veya başarısızlık "
    "iddiası kurulamaz.",
    "",
    "## 4. Bilimsel karar",
    "",
    "**Mevcut Mac M4, FP32, batch=1, CPU giriş/çıkış "
    "koşullarında test edilen ConvNeXt hibritleri "
    "için Full MPS'ye karşı bağımsız holdout ile "
    "doğrulanmış tutarlı bir latency hızlanması "
    "gösterilememiştir.**",
    "",
    "Bu negatif doğrulama sonucu, scheduler "
    "araştırmasında tek bir pilotun minimum latency "
    "değerini ödül veya etiket olarak kullanmanın "
    "riskini somutlaştırır.",
    "",
    "## 5. Kaynak kampanyalar",
    "",
    "Her kampanyadaki mevcut SHA256 kayıtları "
    "yeniden doğrulanmıştır. Bütün dosyaların "
    "yeni SHA256 envanteri `source_inventory.csv` "
    "içindedir.",
    "",
])

for campaign, path in SOURCES.items():
    lines.append(f"- `{campaign}`: `{path}`")

lines.extend([
    "",
    "Yeniden hesaplanan karşılaştırma satırları "
    "`verified_findings.csv` dosyasındadır.",
    "",
    "## 6. Sonraki araştırma yönü",
    "",
    "Yeni donanımlarda aynı workload ve cihaz-geçiş "
    "tanımları korunarak bağımsız tekrarlı latency "
    "kampanyaları çalıştırılmalı; ardından enerji, "
    "bellek sınırı, batch boyutu, gerçek edge "
    "cihaz ayrımı ve scheduling politikaları "
    "ayrı deney değişkenleri olarak eklenmelidir.",
])

inventory_path = OUT / "source_inventory.csv"
findings_path = OUT / "verified_findings.csv"
report_path = OUT / "scientific_summary.md"
metadata_path = OUT / "metadata.json"

savecsv(
    inventory_path,
    [
        "campaign", "relative_path", "bytes",
        "sha256", "listed_in_source_checksums",
        "csv_data_rows",
    ],
    source_inventory
)

savecsv(
    findings_path, FINDING_FIELDS, findings
)

report_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()
except Exception:
    commit = None

metadata = {
    "protocol": "T13_research_synthesis_v1",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "git_head_at_generation": commit,
    "source_campaigns": SOURCES,
    "campaign_count": len(SOURCES),
    "source_file_count": len(source_inventory),
    "verified_source_checksum_entries": verified_checksums,
    "finding_rows": len(findings),
    "raw_data_retimed": False,
    "original_sources_modified": False,
    "holdout_images_excluded_from_prior": 16,
    "t12_one_thread_round_medians": t12_rounds[1],
    "t12_four_thread_round_medians": t12_rounds[4],
    "t12_favorable_image_counts": t12_images,
    "source_scope": (
        "Mac M4 T05–T12 campaigns; source hashes "
        "and paired comparisons independently rechecked"
    ),
}

metadata_path.write_text(
    json.dumps(metadata, indent=2)
)

outputs = [
    Path(__file__),
    inventory_path,
    findings_path,
    report_path,
    metadata_path,
]

checksums_path = OUT / "checksums.json"
checksums_path.write_text(
    json.dumps(
        {p.name: sha(p) for p in outputs},
        indent=2
    )
)

for name, expected in json.loads(
    checksums_path.read_text()
).items():
    assert sha(OUT / name) == expected

assert len(loadcsv(inventory_path)) == len(
    source_inventory
)
assert len(loadcsv(findings_path)) == 38

print("\n=== T13 SYNTHESIS ===")
print("SOURCE CAMPAIGNS:", len(SOURCES))
print("INDEXED SOURCE FILES:", len(source_inventory))
print("VERIFIED CHECKSUM ENTRIES:", verified_checksums)
print("RECOMPUTED FINDINGS:", len(findings))
print("HOLDOUT RESULT: T11 ADVANTAGE NOT REPLICATED")
print("\nT13 SCIENTIFIC SYNTHESIS: PASS")
print("RESULT DIRECTORY:", OUT)
print("REPORT:", report_path)
