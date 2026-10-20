import csv
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3/production_campaigns"
G2 = BASE / "g2_cut_graph_mac"
DEST = BASE / "g3_graph_dataset_mac_v1"
DB = DEST / "graph_dataset.sqlite"
MANIFESTS = ROOT / "configs/partition_manifests_v2"

assert not DB.exists(), "Existing graph dataset will not be overwritten"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def provenance(conn, path, role):
    path = path.resolve()
    assert path.is_file(), path
    conn.execute("""
        INSERT OR IGNORE INTO source_files
        (path, role, sha256, size_bytes)
        VALUES (?, ?, ?, ?)
    """, (
        str(path.relative_to(ROOT)),
        role, sha(path), path.stat().st_size
    ))

MANIFEST_MAP = {
    "resnet50": "resnet50__stage6.json",
    "convnext_base": "convnext_base__fine41.json",
    "convnext_large": "convnext_large__fine41.json",
}

REGISTRY = G2 / "cut_registry_144.csv"
assert REGISTRY.is_file()
with REGISTRY.open(newline="") as f:
    registry = list(csv.DictReader(f))

assert len(registry) == 144
models = sorted({r["model"] for r in registry})
assert len(models) == 6

# Cut definitions are derived from the actual stage boundaries.
# They are not assertions that all possible cuts were executed.
by_model = {
    m: sorted(
        (r for r in registry if r["model"] == m),
        key=lambda r: int(r["cut_after_stage"])
    )
    for m in models
}

conn = sqlite3.connect(DB)
conn.execute("PRAGMA foreign_keys=ON")

try:
    conn.executescript("""
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE source_files (
        path TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL
    );

    CREATE TABLE models (
        model_id TEXT PRIMARY KEY,
        hardware_id TEXT NOT NULL,
        stage_count INTEGER NOT NULL,
        granularity TEXT NOT NULL,
        graph_definition_status TEXT NOT NULL,
        manifest_source TEXT
    );

    CREATE TABLE stage_nodes (
        model_id TEXT NOT NULL,
        stage_index INTEGER NOT NULL,
        stage_id TEXT NOT NULL,
        stage_name TEXT NOT NULL,
        module_paths_json TEXT,
        PRIMARY KEY (model_id, stage_index),
        FOREIGN KEY (model_id) REFERENCES models(model_id)
    );

    CREATE TABLE graph_edges (
        model_id TEXT NOT NULL,
        edge_index INTEGER NOT NULL,
        from_stage_index INTEGER NOT NULL,
        to_stage_index INTEGER NOT NULL,
        edge_kind TEXT NOT NULL,
        PRIMARY KEY (model_id, edge_index),
        FOREIGN KEY (model_id, from_stage_index)
          REFERENCES stage_nodes(model_id, stage_index),
        FOREIGN KEY (model_id, to_stage_index)
          REFERENCES stage_nodes(model_id, stage_index)
    );

    CREATE TABLE cut_boundaries (
        model_id TEXT NOT NULL,
        cut_after_stage INTEGER NOT NULL,
        edge_index INTEGER NOT NULL,
        prior_t26_coverage TEXT NOT NULL,
        prior_t26_cut_alias TEXT NOT NULL,
        prior_coverage_validation TEXT NOT NULL,
        PRIMARY KEY (model_id, cut_after_stage),
        FOREIGN KEY (model_id, edge_index)
          REFERENCES graph_edges(model_id, edge_index)
    );

    CREATE TABLE pilot_results (
        pilot_id TEXT PRIMARY KEY,
        source_campaign TEXT NOT NULL,
        model_id TEXT NOT NULL,
        precision TEXT NOT NULL,
        direction TEXT,
        cut_after_stage INTEGER NOT NULL,
        numerical_status TEXT NOT NULL,
        finite INTEGER,
        strict_allclose INTEGER,
        agreement_value REAL,
        agreement_metric TEXT,
        kl_nats REAL,
        max_abs_error REAL,
        logical_activation_bytes INTEGER,
        is_concurrent_pipeline INTEGER NOT NULL DEFAULT 0,
        is_power_measurement INTEGER NOT NULL DEFAULT 0,
        raw_result_path TEXT NOT NULL,
        raw_result_sha256 TEXT NOT NULL,
        FOREIGN KEY (model_id, cut_after_stage)
          REFERENCES cut_boundaries(model_id, cut_after_stage)
    );
    """)

    conn.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("schema_version", "g3_graph_dataset_v1")
    )
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("measurement_import_status", "g2_pilots_only")
    )
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("graph_semantics", "ordered_module_groups_with_candidate_cuts")
    )

    provenance(conn, REGISTRY, "g2_cut_registry")

    for model, cuts in by_model.items():
        count = int(cuts[0]["stage_count"])
        assert len(cuts) == count - 1
        assert [int(r["cut_after_stage"]) for r in cuts] == list(
            range(1, count)
        )

        manifest_file = MANIFEST_MAP.get(model)
        groups = None
        granularity = "transformer_block"
        graph_status = "registry_defined_not_fully_validated"
        source_path = None

        if manifest_file:
            path = MANIFESTS / manifest_file
            manifest = json.loads(path.read_text())
            assert manifest["model_id"] == model
            assert manifest["group_count"] == count
            assert len(manifest["groups"]) == count
            groups = manifest["groups"]
            granularity = manifest["granularity"]
            graph_status = manifest["execution_status"]
            source_path = str(path.relative_to(ROOT))
            provenance(conn, path, "partition_manifest")

        conn.execute(
            "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?)",
            (
                model, "mac_m4", count, granularity,
                graph_status, source_path
            )
        )

        for index in range(1, count + 1):
            if groups is not None:
                group = groups[index - 1]
                stage_id = group["group_id"]
                stage_name = group["name"]
                modules = group["modules"]
            else:
                if index == 1:
                    stage_id = cuts[0]["stage_before_id"]
                    stage_name = cuts[0]["stage_before_name"]
                else:
                    stage_id = cuts[index - 2]["stage_after_id"]
                    stage_name = cuts[index - 2]["stage_after_name"]
                modules = [stage_name]

            if index < count:
                assert stage_id == cuts[index - 1]["stage_before_id"]
                assert stage_name == cuts[index - 1]["stage_before_name"]
            if index > 1:
                assert stage_id == cuts[index - 2]["stage_after_id"]
                assert stage_name == cuts[index - 2]["stage_after_name"]

            conn.execute(
                "INSERT INTO stage_nodes VALUES (?, ?, ?, ?, ?)",
                (
                    model, index, stage_id,
                    stage_name, json.dumps(modules)
                )
            )

        for cut in cuts:
            k = int(cut["cut_after_stage"])
            conn.execute(
                "INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?)",
                (model, k, k, k + 1, "candidate_cut_boundary")
            )

            coverage = cut["prior_t26_coverage"]
            alias = cut["prior_t26_cut_alias"]
            if model.startswith("convnext") and coverage == "measured":
                expected = {
                    1: "T26B macro cut 1",
                    8: "T26B macro cut 3",
                    40: "T26B macro cut 5",
                }
                assert k in expected, (model, k)
                assert alias == expected[k], (model, k, alias)

            conn.execute(
                "INSERT INTO cut_boundaries VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model, k, k, coverage, alias,
                    "worker_cut_mapping_verified"
                    if model.startswith("convnext") and coverage == "measured"
                    else (
                        "registry_claim_not_reimported"
                        if coverage == "measured"
                        else "not_applicable"
                    )
                )
            )

    def insert_pilot(
        pilot_id, campaign, obj, path,
        agreement_metric, summary_item
    ):
        model = obj["model"]
        precision = obj["precision"]

        expected_cuts = {
            "resnet50": 2,
            "convnext_base": 20,
            "convnext_large": 20,
            "distilgpt2": 1,
            "qwen2_5_1_5b": 10,
            "qwen3_1_7b": 10,
        }

        cut = expected_cuts[model]

        for record in (obj, summary_item):
            for field in (
                "cut", "cut_after_stage", "cut_after_group"
            ):
                if record.get(field) is not None:
                    assert int(record[field]) == cut, (
                        "Pilot cut mismatch",
                        model, precision, field,
                        record[field], cut
                    )

        def lookup(*names):
            # Prefer original per-case measurements.
            # Fall back to the campaign summary only
            # when the original field is absent.
            for record in (obj, summary_item):
                for name in names:
                    if record.get(name) is not None:
                        return record[name]
            return None

        raw_status = (
            obj.get("status")
            or obj.get("numerical_status")
        )

        summary_status = (
            summary_item.get("status")
            or summary_item.get("numerical_status")
        )

        if raw_status and summary_status:
            assert raw_status == summary_status, (
                "Conflicting numerical statuses",
                model, precision,
                raw_status, summary_status
            )

        status = raw_status or summary_status
        assert isinstance(status, str) and status, (
            "Numerical status missing from both sources",
            model, precision,
            list(obj), list(summary_item)
        )

        finite = lookup("finite", "output_finite")
        strict = lookup(
            "strict_allclose_1e_3",
            "strict_allclose",
            "allclose",
        )

        if agreement_metric == "top1_agreement":
            agreement = lookup(
                "top1_agreement",
                "top1_match_rate",
                "top1",
            )
        elif agreement_metric == "token_agreement":
            agreement = lookup("token_agreement")
        else:
            raise ValueError(agreement_metric)

        kl = lookup("kl_fp32_to_candidate_nats")

        max_abs = lookup(
            "max_abs_logit_error",
            "max_abs_error",
            "max_abs",
        )

        activation_bytes = lookup(
            "activation_bytes",
            "logical_activation_bytes",
        )

        conn.execute(
            '''
            INSERT INTO pilot_results (
                pilot_id,
                source_campaign,
                model_id,
                precision,
                direction,
                cut_after_stage,
                numerical_status,
                finite,
                strict_allclose,
                agreement_value,
                agreement_metric,
                kl_nats,
                max_abs_error,
                logical_activation_bytes,
                is_concurrent_pipeline,
                is_power_measurement,
                raw_result_path,
                raw_result_sha256
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, 0, 0, ?, ?
            )
            ''',
            (
                pilot_id,
                campaign,
                model,
                precision,
                lookup("direction") or "cpu_mps",
                cut,
                status,
                None if finite is None else int(finite),
                None if strict is None else int(strict),
                agreement,
                agreement_metric,
                kl,
                max_abs,
                activation_bytes,
                str(path.relative_to(ROOT)),
                sha(path),
            )
        )

        provenance(conn, path, "g2_pilot_result")

        print(
            "IMPORTED:",
            model, precision,
            "cut=", cut,
            "status=", status,
            "agreement=", agreement,
            flush=True,
        )

    vision = G2 / "vision_pilot_v1"
    text = G2 / "text_pilot_v1"

    for folder, campaign, metric in (
        (vision, "g2_vision_pilot_v1", "top1_agreement"),
        (text, "g2_text_pilot_v1", "token_agreement"),
    ):
        summary = folder / "summary.json"
        assert summary.is_file(), summary
        provenance(conn, summary, "g2_pilot_summary")
        parsed = json.loads(summary.read_text())

        # Accept a list summary or a dictionary carrying case entries.
        if isinstance(parsed, list):
            cases = parsed
        else:
            cases = parsed.get("cases", parsed.get("results"))
        assert isinstance(cases, list), summary

        imported = 0
        for item in cases:
            model = item.get("model")
            precision = item.get("precision")
            assert model and precision

            candidates = [
                folder / f"{model}_{precision}.json",
                folder / f"{model}_cut2_{precision}.json",
                folder / f"{model}_cut20_{precision}.json",
            ]
            raw = next((p for p in candidates if p.is_file()), None)
            if raw is None:
                # Vision pilot may use a more descriptive file name.
                possibilities = sorted(folder.rglob(
                    f"*{model}*{precision}*.json"
                ))
                raw = next(
                    (p for p in possibilities if p.name != "summary.json"),
                    None
                )
            assert raw is not None, (campaign, model, precision)

            obj = json.loads(raw.read_text())
            assert obj["model"] == model
            assert obj["precision"] == precision
            insert_pilot(
                f"{campaign}:{model}:{precision}",
                campaign, obj, raw, metric, item
            )
            imported += 1

        assert imported == 6, (campaign, imported)

    assert conn.execute(
        "SELECT COUNT(*) FROM stage_nodes"
    ).fetchone()[0] == 150

    assert conn.execute(
        "SELECT COUNT(*) FROM graph_edges"
    ).fetchone()[0] == 144

    assert conn.execute(
        "SELECT COUNT(*) FROM pilot_results"
    ).fetchone()[0] == 12

    assert conn.execute(
        "SELECT COUNT(*) FROM cut_boundaries "
        "WHERE prior_t26_coverage='measured'"
    ).fetchone()[0] == 17

    assert conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall() == []

    conn.commit()
    assert conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    print("\nG3 GRAPH SQLITE: PASS")
    for table in (
        "models", "stage_nodes", "graph_edges",
        "cut_boundaries", "pilot_results", "source_files"
    ):
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        print(table, count)

    print("OUTPUT:", DB)
    print("OLD SQLITE MODIFIED: NO")
    print("BENCHMARK RERUN: NO")

except BaseException:
    conn.rollback()
    conn.close()
    if DB.exists():
        DB.unlink()
    raise
else:
    conn.close()
