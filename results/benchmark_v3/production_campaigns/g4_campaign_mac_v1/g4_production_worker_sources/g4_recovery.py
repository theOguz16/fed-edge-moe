import argparse
import hashlib
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path.cwd().resolve()
RUN = (
    ROOT / "results/benchmark_v3/production_campaigns"
    / "g4_campaign_mac_v1/numerical_v1"
)
DB = RUN / "numerical.sqlite"

FINISHED = {
    "numerical_pass",
    "numerical_mismatch",
    "numerical_invalid",
}
KNOWN = FINISHED | {
    "pending",
    "excluded_not_run",
    "execution_failed",
}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    assert DB.is_file(), DB
    assert not (RUN / ".runner.lock").exists(), (
        "A runner lock exists. Do not modify the "
        "queue while a campaign may be running."
    )

    uri = f"file:{DB}?mode={'rw' if args.retry_failed else 'ro'}"
    con = sqlite3.connect(uri, uri=True)

    try:
        assert con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        rows = con.execute("""
            SELECT case_id, status, attempts,
                   result_path, result_sha256
            FROM cases
        """).fetchall()

        assert len(rows) == 576
        assert set(row[1] for row in rows) <= KNOWN

        counts = Counter()
        incomplete_attempts = 0

        for case_id, status, attempts, path, digest in rows:
            counts[status] += 1

            if status in FINISHED:
                assert path and digest, case_id
                file = ROOT / path
                assert file.is_file(), file
                assert sha(file) == digest, case_id

            elif status == "pending":
                assert not path and not digest, case_id
                if attempts > 0:
                    incomplete_attempts += 1

            elif status == "excluded_not_run":
                assert not path and not digest, case_id

        print("G4 RECOVERY AUDIT: PASS")
        for status in sorted(counts):
            print(status, counts[status])

        print("PENDING WITH PRIOR ATTEMPTS:",
              incomplete_attempts)

        failures = counts["execution_failed"]

        if args.retry_failed:
            con.execute("BEGIN IMMEDIATE")
            updated = con.execute("""
                UPDATE cases
                SET status='pending',
                    note='Explicit retry requested after failure'
                WHERE status='execution_failed'
                  AND result_path IS NULL
                  AND result_sha256 IS NULL
            """).rowcount
            assert updated == failures
            con.commit()
            print("EXPLICITLY REQUEUED:", updated)
        else:
            print("DATABASE MODIFIED: NO")

        print("FINISHED CASES REQUEUED: 0")

    finally:
        con.close()

if __name__ == "__main__":
    main()
