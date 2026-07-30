"""Audit the index against what is actually on disk.

The original check_results.py existed because a benchmark Job could fail on the
cluster while run.sh still exited 0, leaving entries marked "success" with no
result file behind them. The exit code is no longer smuggled through a shell
script, but an audit is still worth having: a pull can be interrupted, a run
directory can be deleted, and `--resume` trusts the index.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import ingest, settings, store, util
from .store import RunRecord


def _problems(record: RunRecord) -> list[str]:
    issues = []
    if not record.workloads:
        issues.append("no workloads recorded")
    for workload in record.workloads:
        relative = (record.results or {}).get(workload)
        if not relative:
            issues.append(f"{workload}: no result file recorded")
            continue
        path = settings.REPO / relative
        if not path.exists():
            issues.append(f"{workload}: {relative} is missing")
        elif not ingest.result_is_usable(path):
            issues.append(f"{workload}: {relative} is empty, truncated or not a report")
    issues += _empty_measurements(record)
    return issues


def _empty_measurements(record: RunRecord) -> list[str]:
    """Reports that exist but measured nothing — a success in name only.

    Reads the cached summary rather than re-parsing ~100 MB files, so this only
    reports on runs that have been loaded at least once (by `bench report`).
    """
    if not record.run_dir:
        return []
    cache_path = settings.REPO / record.run_dir / ingest.CACHE_NAME
    if not cache_path.exists():
        return []
    try:
        cached = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    issues = []
    for workload, entry in sorted((cached.get("workloads") or {}).items()):
        rows = entry.get("rows") or []
        if rows and not any((row.get("requests_successful") or 0) > 0 for row in rows):
            issues.append(f"{workload}: report contains zero successful requests")
    return issues


def run(*, fix: bool, index_path: Path = settings.INDEX_PATH) -> int:
    records = store.latest_by_tag(index_path)
    if not records:
        util.log(f"no runs recorded yet ({index_path} is empty or absent)")
        return 0

    clean: list[str] = []
    broken: list[tuple[RunRecord, list[str]]] = []
    for record in sorted(records.values(), key=lambda r: (r.experiment, r.tag)):
        issues = _problems(record)
        if issues and record.status == store.SUCCESS:
            broken.append((record, issues))
        elif not issues:
            clean.append(record.tag)

    by_status: dict[str, int] = {}
    for record in records.values():
        by_status[record.status] = by_status.get(record.status, 0) + 1

    util.rule("run index")
    print(
        util.table(
            ["status", "runs"],
            [[status, count] for status, count in sorted(by_status.items())],
        )
    )

    if broken:
        print()
        util.rule("runs marked success whose results are unusable")
        print(
            util.table(
                ["tag", "experiment", "problem"],
                [[r.tag, r.experiment, "; ".join(i)] for r, i in broken],
            )
        )
    else:
        print()
        util.ok(f"{len(clean)} run(s) have complete, parseable results")

    orphans = _orphan_dirs(records)
    if orphans:
        print()
        util.warn(
            f"{len(orphans)} run directory(ies) exist with no index entry: "
            f"{', '.join(orphans[:8])}"
        )

    if not fix:
        if broken:
            print()
            util.log("rerun `./bench doctor --fix` to mark these for a re-run, then:")
            util.log("  ./bench run <experiment> --resume")
        return 1 if broken else 0

    for record, issues in broken:
        corrected = store.RunRecord.from_json(record.to_json())
        corrected.status = store.FAILED
        corrected.failure_kind = "audit"
        corrected.failure_detail = "; ".join(issues)[:500]
        corrected.finished_at = store.utc_now()
        corrected.note = "corrected by `bench doctor --fix`: recorded success without usable results"
        store.append(corrected, index_path=index_path)
    print()
    util.ok(f"appended {len(broken)} corrected entry(ies) to {index_path}")
    util.log("now run: ./bench run <experiment> --resume")
    return 0


def _orphan_dirs(records: dict[str, RunRecord]) -> list[str]:
    if not settings.RUNS_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in settings.RUNS_DIR.iterdir()
        if path.is_dir() and path.name not in records
    )
