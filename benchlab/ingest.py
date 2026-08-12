"""Turn stored GuideLLM result JSONs into one tidy DataFrame.

One row per (run, workload, load level):
  * a steady-rate workload contributes one row;
  * a concurrency ladder contributes one row per level (8/16/32/64), with the
    level read from the scenario rather than hardcoded.

Everything downstream (plots, comparisons) reads this frame, so
there is a single definition of every metric column.

A single GuideLLM result file can be ~100 MB, and a sweep has dozens, so the
handful of numbers we actually plot are cached in each run's summary.json,
fingerprinted by size and mtime. Reporting on a finished sweep then costs
milliseconds instead of parsing gigabytes every time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import settings, store, util, workload_registry
from .store import RunRecord

CACHE_NAME = "summary.json"
CACHE_VERSION = 1

# (GuideLLM metric key, our column prefix)
METRIC_SPECS = [
    ("request_latency", "latency_s"),
    ("request_concurrency", "req_conc"),
    ("time_to_first_token_ms", "ttft_ms"),
    ("time_per_output_token_ms", "tpot_ms"),
    ("inter_token_latency_ms", "itl_ms"),
    ("requests_per_second", "rps"),
    ("prompt_tokens_per_second", "prompt_tps"),
    ("output_tokens_per_second", "output_tps"),
    ("tokens_per_second", "tps"),
]
STAT_SUFFIXES = ["mean", "p50", "p90", "p99"]

METRIC_COLUMNS = [f"{prefix}_{stat}" for _, prefix in METRIC_SPECS for stat in STAT_SUFFIXES]


def _dist(metrics: dict[str, Any], key: str) -> dict[str, float | None]:
    successful = (metrics.get(key) or {}).get("successful") or {}
    percentiles = successful.get("percentiles") or {}
    return {
        "mean": successful.get("mean"),
        "p50": percentiles.get("p50"),
        "p90": percentiles.get("p90"),
        "p99": percentiles.get("p99"),
    }


def _target(benchmark: dict[str, Any], mode: str, index: int, targets: list[float]) -> float:
    strategy = (benchmark.get("config") or {}).get("strategy") or {}
    if mode in {"rate", "sweep"}:
        for key in ("rate", "rate_per_second"):
            if strategy.get(key) is not None:
                return float(strategy[key])
    else:
        for key in ("streams", "concurrency", "max_concurrency", "stream_count", "worker_count"):
            if strategy.get(key) is not None:
                return float(strategy[key])
    if index < len(targets):
        return targets[index]
    return float(index)


def _measurements(benchmark: dict[str, Any], mode: str, index: int, targets: list[float]):
    """Just the numbers: no run metadata, so this is safe to cache."""
    metrics = benchmark.get("metrics") or {}
    strategy = (benchmark.get("config") or {}).get("strategy") or {}
    totals = metrics.get("request_totals") or {}
    errored = totals.get("errored") or 0
    total = totals.get("total") or 0

    row: dict[str, Any] = {
        "target": _target(benchmark, mode, index, targets),
        "strategy_type": str(strategy.get("type_") or strategy.get("kind") or "").lower(),
        "offered_rps": strategy.get("rate", strategy.get("rate_per_second")),
        "duration_s": round(float(benchmark.get("duration") or 0), 1),
        "requests_total": total,
        "requests_successful": totals.get("successful") or 0,
        "requests_errored": errored,
        "requests_incomplete": totals.get("incomplete") or 0,
        "error_rate": round(errored / total, 4) if total else None,
    }
    for metric_key, prefix in METRIC_SPECS:
        stats = _dist(metrics, metric_key)
        for stat in STAT_SUFFIXES:
            row[f"{prefix}_{stat}"] = stats[stat]
    return row


def _fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": CACHE_VERSION, "workloads": {}}
    try:
        cached = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"version": CACHE_VERSION, "workloads": {}}
    if cached.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "workloads": {}}
    cached.setdefault("workloads", {})
    return cached


def _extract(path: Path, mode: str, targets: list[float]) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        util.warn(f"skipping unreadable result {path.name}: {exc}")
        return None
    return [
        _measurements(benchmark, mode, index, targets)
        for index, benchmark in enumerate(payload.get("benchmarks") or [])
    ]


def _workload_shape(name: str) -> tuple[str, list[float]]:
    try:
        workload = workload_registry.load(name)
        return workload.mode, workload.targets
    except SystemExit:
        # A workload can be renamed or retired after runs exist; infer the shape
        # rather than dropping results we already paid for.
        return ("concurrent" if "user" in name or "conc" in name else "rate"), []


def _rows_for(record: RunRecord, *, refresh: bool = False) -> list[dict[str, Any]]:
    """Cached measurements for one run, one dict per load level."""
    files = store.result_files(record)
    if not files:
        return []
    cache_path = settings.REPO / record.run_dir / CACHE_NAME if record.run_dir else None
    cache = _load_cache(cache_path) if cache_path else {"workloads": {}}
    dirty = False

    rows: list[dict[str, Any]] = []
    for workload_name, path in sorted(files.items()):
        mode, targets = _workload_shape(workload_name)
        fingerprint = _fingerprint(path)
        entry = cache["workloads"].get(workload_name)
        cached_rows = (entry.get("rows") or []) if entry else []
        cache_missing_keys = bool(cached_rows) and any(
            (
                "strategy_type" not in row
                or "offered_rps" not in row
                or "req_conc_mean" not in row
                or "req_conc_p50" not in row
                or "req_conc_p99" not in row
            )
            for row in cached_rows[:3]
        )
        if not refresh and entry and entry.get("fingerprint") == fingerprint and not cache_missing_keys:
            measurements = entry.get("rows") or []
        else:
            extracted = _extract(path, mode, targets)
            if extracted is None:
                continue
            measurements = extracted
            cache["workloads"][workload_name] = {
                "fingerprint": fingerprint,
                "mode": mode,
                "rows": measurements,
            }
            dirty = True
        for measurement in measurements:
            rows.append({"workload": workload_name, "mode": mode, **measurement})

    if dirty and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, sort_keys=True))
    return rows


def _record_fields(record: RunRecord) -> dict[str, Any]:
    return {
        "tag": record.tag,
        "experiment": record.experiment,
        "model": record.model,
        "strategy": record.strategy,
        "param": record.param,
        "value": record.value,
        "label": record.label,
        "suffix": record.suffix or "",
        "status": record.status,
        "topology": record.topology.label,
        "gpus": record.topology.gpus,
        "derived_axis": False,
        "axis_context": "",
    }


def _external_axes_from_tag(tag: str) -> list[tuple[str, str]]:
    """Parse external tag convention: param-val-param-val."""
    parts = [p for p in str(tag).split("-") if p]
    if len(parts) < 2 or len(parts) % 2 != 0:
        return []
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]


def _axes_from_label(label: str) -> list[tuple[str, str]]:
    """Parse axis labels like `MAX_NUM_SEQS=64, MAX_NUM_BATCHED_TOKENS=1024`."""
    pairs: list[tuple[str, str]] = []
    for part in str(label or "").split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        axis, value = item.split("=", 1)
        axis = axis.strip()
        value = value.strip()
        if axis and value:
            pairs.append((axis, value))
    return pairs


# Columns that qualify a measurement rather than describe it: two runs sharing an
# axis value are still not the same series if either of these differs.
QUALIFIERS = ("model", "topology")


def label_prefixes(df: pd.DataFrame) -> list[str]:
    """Which qualifier columns a series label in this frame has to name.

    Naming them unconditionally would clutter every legend in the common case of
    one model on one topology; omitting them when they vary would silently plot a
    two-GPU run and a one-GPU run as the same line.
    """
    return [column for column in QUALIFIERS if df[column].nunique() > 1]


def series_label(row: pd.Series, prefixes: list[str], value: str) -> str:
    return ": ".join([*(str(row[column]) for column in prefixes), value])


def frame(records: list[RunRecord], *, quiet: bool = False, refresh: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        measurements = _rows_for(record, refresh=refresh)
        if not measurements:
            missing.append(record.tag)
            continue
        fields = _record_fields(record)
        rows.extend({**fields, **measurement} for measurement in measurements)
        if record.experiment == "external":
            for axis, value in _external_axes_from_tag(record.tag):
                derived = dict(fields)
                derived["param"] = axis
                derived["value"] = value
                derived["label"] = f"{axis}={value}"
                derived["derived_axis"] = True
                rows.extend({**derived, **measurement} for measurement in measurements)
        elif record.strategy == "grid":
            pairs = _axes_from_label(record.label)
            for axis, value in pairs:
                others = [(a, v) for a, v in pairs if a != axis]
                derived = dict(fields)
                derived["param"] = axis
                derived["value"] = value
                derived["label"] = f"{axis}={value}"
                derived["derived_axis"] = True
                derived["axis_context"] = ", ".join(f"{a}={v}" for a, v in others)
                rows.extend({**derived, **measurement} for measurement in measurements)

    if missing and not quiet:
        util.warn(f"{len(missing)} run(s) have no usable results yet: {', '.join(missing[:6])}")
    if not rows:
        util.die("no usable results found — run something first, or check `./bench doctor`")

    df = pd.DataFrame(rows)

    def display(row: pd.Series) -> str:
        # A swept run is named by its axis value, since that is the plot's
        # subject. A run with no axis (a one-off, or an import whose axis was
        # never recorded) has only its tag to go by — 'auto' would be a lie, and
        # a real axis value of 'auto' means something quite different.
        value = str(row["value"])
        if not value or row["param"] in ("", "unknown"):
            value = str(row["tag"])
        return f"{value} [{row['suffix']}]" if row["suffix"] else value

    df["value_display"] = df.apply(display, axis=1)
    prefixes = label_prefixes(df)
    df["series"] = df.apply(
        lambda r: series_label(r, prefixes, r["value_display"]), axis=1
    )
    # Series labels key the plots and their legends, so two different runs must
    # never share one — otherwise they'd be drawn over each other, or dropped as
    # duplicates. Imported one-off runs in particular have uninformative values.
    per_series = df.groupby("series")["tag"].nunique()
    ambiguous = set(per_series[per_series > 1].index)
    if ambiguous:
        df["series"] = df.apply(
            lambda r: f"{r['series']} ({r['tag']})" if r["series"] in ambiguous else r["series"],
            axis=1,
        )
    if not quiet:
        util.step(
            f"loaded {len(df)} rows from {df['tag'].nunique()} run(s), "
            f"{df['workload'].nunique()} workload(s), {df['model'].nunique()} model(s)"
        )
        shapes = sorted(df["topology"].unique())
        if len(shapes) > 1:
            util.step(
                f"this view mixes GPU topologies ({', '.join(shapes)}); each series "
                f"names its own, but a difference between them is a difference of "
                f"hardware shape, not of the swept knob"
            )
    return df


def result_is_usable(path: Path) -> bool:
    """Cheap validity check for a possibly enormous result file.

    Full parsing costs seconds per file, so this looks for the structure a
    complete GuideLLM report has: a JSON object that mentions benchmarks and was
    closed properly. That catches the realistic failures — an empty file, an
    error page, or a report truncated by a full disk.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with open(path, "rb") as handle:
            head = handle.read(1_000_000)
            if not head.lstrip().startswith(b"{"):
                return False
            if b'"benchmarks"' not in head:
                return False
            handle.seek(max(0, path.stat().st_size - 4096))
            return handle.read().rstrip().endswith(b"}")
    except OSError:
        return False


def _mode_from_benchmarks(benchmarks: list[dict[str, Any]]) -> str:
    """Infer workload mode from strategy fields in raw GuideLLM records."""
    has_rate = False
    has_concurrency = False
    for benchmark in benchmarks:
        strategy = ((benchmark or {}).get("config") or {}).get("strategy") or {}
        if strategy.get("rate") is not None or strategy.get("rate_per_second") is not None:
            has_rate = True
        if any(
            strategy.get(key) is not None
            for key in ("streams", "concurrency", "max_concurrency", "stream_count", "worker_count")
        ):
            has_concurrency = True
    if has_rate:
        return "rate"
    if has_concurrency:
        return "concurrent"
    return "rate"


def frame_from_benchmark_json(path: Path, *, workload: str | None = None) -> pd.DataFrame:
    """Parse a standalone GuideLLM benchmarks.json into report-ready rows."""
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        util.die(f"failed to read '{path}': {exc}")
    benchmarks = payload.get("benchmarks") or []
    if not benchmarks:
        util.die(f"'{path}' has no benchmarks to report")

    mode = _mode_from_benchmarks(benchmarks)
    rows = [
        {"workload": workload or path.stem, "mode": mode, **_measurements(benchmark, mode, index, [])}
        for index, benchmark in enumerate(benchmarks)
    ]
    df = pd.DataFrame(rows)
    stem = util.sanitize(path.stem) or "json"
    tag = f"json-{stem}"
    df["tag"] = tag
    df["experiment"] = "json"
    df["model"] = "json-input"
    df["strategy"] = "json"
    df["param"] = "unknown"
    df["value"] = tag
    df["label"] = tag
    df["suffix"] = ""
    df["status"] = "success"
    df["topology"] = "json"
    df["gpus"] = 0
    df["derived_axis"] = False
    df["axis_context"] = ""
    df["value_display"] = tag
    df["series"] = tag
    return df
