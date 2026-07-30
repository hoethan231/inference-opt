#!/usr/bin/env python3
"""
load_results.py - the single place that turns pulled GuideLLM result JSONs
(results/outputs/vllm-{rate,users}-<tag>.json) plus the sweep manifest into a
tidy pandas DataFrame for plotting/comparison.

This used to be a standalone `aggregate.py` that wrote results/aggregated.csv;
now `plot_results.py` and `compare_tags.py` just import `load_results()` and
work off the JSON files directly, so there's no intermediate CSV to keep in
sync.

One row per (tag, test_type, load-level):
  - "rate" results have 1 row (single Poisson rate).
  - "concurrent" ("users") results have up to 4 rows, one per stream level
    (8/16/32/64 from scenarios/coding-agent-users.json).

Columns: tag, param, value, suffix, value_display, test_type, target,
request outcome counts + error_rate, and mean/p50/p90/p99 for the latency and
throughput metrics in METRIC_SPECS below.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

sys.path.insert(0, str(HERE))
from grid import AUTO_LABEL  # noqa: E402

DEFAULT_RESULTS_DIR = REPO / "results" / "outputs"
DEFAULT_MANIFEST = HERE / "manifest.jsonl"
DEFAULT_SCENARIOS_DIR = REPO / "scenarios"

FNAME_RE = re.compile(r"^vllm-(rate|users)-(.+)\.json$")

# (guidellm metric key, output column prefix)
METRIC_SPECS = [
    ("request_latency", "request_latency_s"),
    ("time_to_first_token_ms", "ttft_ms"),
    ("time_per_output_token_ms", "tpot_ms"),
    ("inter_token_latency_ms", "itl_ms"),
    ("requests_per_second", "requests_per_second"),
    ("prompt_tokens_per_second", "prompt_tokens_per_second"),
    ("output_tokens_per_second", "output_tokens_per_second"),
    ("tokens_per_second", "tokens_per_second"),
]
STAT_SUFFIXES = ["mean", "p50", "p90", "p99"]


def load_manifest(manifest_path=DEFAULT_MANIFEST):
    """tag -> most recent manifest record (prefers a 'success' entry)."""
    records = {}
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return records
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prev = records.get(rec["tag"])
            if prev is None or rec.get("status") == "success":
                records[rec["tag"]] = rec
    return records


def load_scenario_streams(scenarios_dir=DEFAULT_SCENARIOS_DIR):
    path = Path(scenarios_dir) / "coding-agent-users.json"
    try:
        with open(path) as f:
            spec = json.load(f)
        return spec["spec"]["profile"]["streams"]
    except Exception:
        return []


def _dist_stats(metrics, key):
    d = (metrics.get(key) or {}).get("successful") or {}
    pct = d.get("percentiles") or {}
    return {
        "mean": d.get("mean"),
        "p50": pct.get("p50"),
        "p90": pct.get("p90"),
        "p99": pct.get("p99"),
    }


def _get_target(test_type, bench_config, index, scenario_streams):
    strategy = (bench_config or {}).get("strategy") or {}
    if test_type == "rate":
        return strategy.get("rate")
    for k in ("streams", "concurrency", "max_concurrency", "stream_count", "worker_count"):
        v = strategy.get(k)
        if v is not None:
            return v
    if index < len(scenario_streams):
        return scenario_streams[index]
    return index


def _build_row(tag, test_type, target, bench, manifest_rec):
    metrics = bench.get("metrics") or {}
    totals = (metrics.get("request_totals") or {}).get
    successful = totals("successful") or 0
    errored = totals("errored") or 0
    incomplete = totals("incomplete") or 0
    total = totals("total") or 0

    rec = manifest_rec or {}
    row = {
        "tag": tag,
        "param": rec.get("param", ""),
        "value": rec.get("value", ""),
        "suffix": rec.get("suffix", "") or "",
        "test_type": test_type,
        "target": target,
        "duration_s": round(bench.get("duration") or 0, 1),
        "warmup_duration_s": round(bench.get("warmup_duration") or 0, 1),
        "requests_total": total,
        "requests_successful": successful,
        "requests_errored": errored,
        "requests_incomplete": incomplete,
        "error_rate": round(errored / total, 4) if total else None,
    }
    for metric_key, prefix in METRIC_SPECS:
        stats = _dist_stats(metrics, metric_key)
        for suffix in STAT_SUFFIXES:
            row["{}_{}".format(prefix, suffix)] = stats.get(suffix)
    return row


def load_results(
    results_dir=DEFAULT_RESULTS_DIR,
    manifest_path=DEFAULT_MANIFEST,
    scenarios_dir=DEFAULT_SCENARIOS_DIR,
    quiet=False,
):
    """Read every results/outputs/vllm-{rate,users}-<tag>.json and return a
    tidy DataFrame (one row per tag/test_type/load-level)."""
    results_dir = Path(results_dir)
    manifest = load_manifest(manifest_path)
    scenario_streams = load_scenario_streams(scenarios_dir)

    files = sorted(results_dir.glob("vllm-rate-*.json")) + sorted(
        results_dir.glob("vllm-users-*.json")
    )
    if not files:
        sys.exit("no vllm-{{rate,users}}-*.json result files found in {}".format(results_dir))

    rows = []
    for path in files:
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        kind, tag = m.group(1), m.group(2)
        test_type = "rate" if kind == "rate" else "concurrent"
        manifest_rec = manifest.get(tag)
        if manifest_rec is None and not quiet:
            print("warning: no manifest entry for tag '{}'".format(tag))
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                print("warning: failed to parse {}: {}".format(path.name, exc))
            continue
        for i, bench in enumerate(data.get("benchmarks") or []):
            target = _get_target(test_type, bench.get("config"), i, scenario_streams)
            rows.append(_build_row(tag, test_type, target, bench, manifest_rec))

    if not rows:
        sys.exit("parsed 0 benchmark rows out of {} files".format(len(files)))

    df = pd.DataFrame(rows)
    # A human-friendly value label: the swept value, disambiguated by suffix
    # when one was used (so re-runs of the same value under a custom suffix
    # show up as distinct series in the plots).
    def _display(r):
        base = AUTO_LABEL if (pd.isna(r["value"]) or r["value"] == "") else r["value"]
        return "{} [{}]".format(base, r["suffix"]) if r["suffix"] else base

    df["value_display"] = df.apply(_display, axis=1)
    if not quiet:
        n_tags = df["tag"].nunique()
        print("Loaded {} rows across {} tag(s) from {}".format(len(df), n_tags, results_dir))
    return df


if __name__ == "__main__":
    # Quick sanity check / preview when run directly.
    frame = load_results()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame[["tag", "param", "value", "suffix", "test_type", "target",
                     "error_rate", "ttft_ms_p50", "tokens_per_second_mean"]].to_string(index=False))
