#!/usr/bin/env python3
"""
aggregate.py - collects all pulled GuideLLM result JSONs + the sweep manifest
into one flat CSV for analysis (pivot table / spreadsheet / pandas).

One row per (config, test_type, load-level):
  - "rate" results have 1 row (single Poisson rate).
  - "concurrent" ("users") results have up to 4 rows, one per stream level
    (8/16/32/64 from scenarios/coding-agent-users.json).

Usage:
    python3 sweep/aggregate.py
    python3 sweep/aggregate.py --results-dir ./results --out results/aggregated.csv
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

sys.path.insert(0, str(HERE))
from grid import BASELINE

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

CONFIG_COLUMNS = list(BASELINE.keys())


def load_manifest(manifest_path):
    """tag -> most recent manifest record (prefers a 'success' entry)."""
    records = {}
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


def load_scenario_streams(scenarios_dir):
    path = scenarios_dir / "coding-agent-users.json"
    try:
        with open(path) as f:
            spec = json.load(f)
        return spec["spec"]["profile"]["streams"]
    except Exception:
        return []


def dist_stats(metrics, key):
    d = (metrics.get(key) or {}).get("successful") or {}
    pct = d.get("percentiles") or {}
    return {
        "mean": d.get("mean"),
        "p50": pct.get("p50"),
        "p90": pct.get("p90"),
        "p99": pct.get("p99"),
    }


def get_target(test_type, bench_config, index, scenario_streams):
    strategy = (bench_config or {}).get("strategy") or {}
    if test_type == "rate":
        return strategy.get("rate"), strategy
    for k in ("streams", "concurrency", "max_concurrency", "stream_count", "worker_count"):
        v = strategy.get(k)
        if v is not None:
            return v, strategy
    if index < len(scenario_streams):
        return scenario_streams[index], strategy
    return index, strategy


def effective_config(overrides):
    cfg = dict(BASELINE)
    cfg.update(overrides or {})
    return cfg


def build_row(tag, test_type, target, bench, manifest_rec, strategy):
    metrics = bench.get("metrics") or {}
    totals = (metrics.get("request_totals") or {}).get
    successful = totals("successful") or 0
    errored = totals("errored") or 0
    incomplete = totals("incomplete") or 0
    total = totals("total") or 0

    overrides = (manifest_rec or {}).get("overrides", {})
    cfg = effective_config(overrides)

    row = {
        "tag": tag,
        "param": (manifest_rec or {}).get("param", ""),
        "value": (manifest_rec or {}).get("value", ""),
        "test_type": test_type,
        "target": target,
    }
    row.update({("cfg_" + k): v for k, v in cfg.items()})
    row.update(
        {
            "duration_s": round(bench.get("duration") or 0, 1),
            "warmup_duration_s": round(bench.get("warmup_duration") or 0, 1),
            "requests_total": total,
            "requests_successful": successful,
            "requests_errored": errored,
            "requests_incomplete": incomplete,
            "error_rate": round(errored / total, 4) if total else None,
        }
    )
    for metric_key, prefix in METRIC_SPECS:
        stats = dist_stats(metrics, metric_key)
        for suffix in STAT_SUFFIXES:
            row["{}_{}".format(prefix, suffix)] = stats.get(suffix)
    row["raw_strategy"] = json.dumps(strategy)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--manifest", default=str(HERE / "manifest.jsonl"))
    ap.add_argument("--scenarios-dir", default=str(REPO / "scenarios"))
    ap.add_argument("--out", default=str(REPO / "results" / "aggregated.csv"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    manifest = load_manifest(Path(args.manifest))
    scenario_streams = load_scenario_streams(Path(args.scenarios_dir))

    files = sorted(results_dir.glob("vllm-*.json"))
    if not files:
        sys.exit("no result JSON files found in {}".format(results_dir))

    rows = []
    for path in files:
        m = FNAME_RE.match(path.name)
        if not m:
            print("skipping unrecognized file: {}".format(path.name))
            continue
        kind, tag = m.group(1), m.group(2)
        test_type = "rate" if kind == "rate" else "concurrent"
        manifest_rec = manifest.get(tag)
        if manifest_rec is None:
            print("warning: no manifest entry for tag '{}' "
                  "(config columns will show baseline-only values)".format(tag))

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print("warning: failed to parse {}: {}".format(path.name, exc))
            continue

        for i, bench in enumerate(data.get("benchmarks") or []):
            target, strategy = get_target(test_type, bench.get("config"), i, scenario_streams)
            rows.append(build_row(tag, test_type, target, bench, manifest_rec, strategy))

    if not rows:
        sys.exit("parsed 0 benchmark rows out of {} files".format(len(files)))

    fieldnames = list(rows[0].keys())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote {} rows ({} files) -> {}".format(len(rows), len(files), out_path))
    n_missing_manifest = sum(1 for p in files if FNAME_RE.match(p.name)
                              and FNAME_RE.match(p.name).group(2) not in manifest)
    if n_missing_manifest:
        print("({} file(s) had no matching manifest entry)".format(n_missing_manifest))


if __name__ == "__main__":
    main()
