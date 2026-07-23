#!/usr/bin/env python3
"""
plot_results.py - sweep result plots + "best config" recommendation.

Reads the pulled GuideLLM result JSONs directly (via load_results.py, which
scans results/outputs/vllm-{rate,users}-<tag>.json + sweep/manifest.jsonl) --
there is no intermediate aggregated.csv anymore.

By default it renders the whole OFAT sweep:

  1. Overview charts with every tag side by side (rate + concurrent).
  2. A latency-vs-throughput frontier across all configs.
  3. One detail chart per swept parameter (bar chart at the steady rate test,
     line chart across 8/16/32/64 concurrent users).
  4. A per-parameter leaderboard + a "recommended" config combining the best
     independently-tested value per parameter.

Or plot just one parameter's sweep with --param:

    python3 sweep/plot_results.py                       # everything
    python3 sweep/plot_results.py --param gpu_memory_utilization
    python3 sweep/plot_results.py --param max-num-seqs  # sanitized name / substring ok

Custom-suffixed runs (run_sweep.py --suffix <s>) show up as distinct series
labelled "<value> [<suffix>]", so you can plot re-runs next to the originals.

IMPORTANT CAVEAT: this sweep is one-factor-at-a-time (OFAT), not a full grid
search -- the "recommended" config is the best *independent* value per
parameter, a reasonable starting guess, not a proven joint optimum. Treat the
final rerun as a confirmation step.
"""
import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; plots are saved as PNG files below
import matplotlib.pyplot as plt
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from grid import BASELINE, SWEEP, display_value, sanitize  # noqa: E402
from load_results import load_results  # noqa: E402

OUT_DIR = REPO / "results" / "plots"
BASE_PARAMS_FILE = REPO / "config" / "vllm-params.yaml"
RECOMMENDED_PARAMS_FILE = REPO / "config" / "generated" / "vllm-params.recommended.yaml"

# Rows with no manifest match (stray/duplicate runs); dropped from the
# comparison views so they don't skew things.
DROP_TAGS = {"20260716-203022"}

# (column, "min" or "max") - direction of "better" for the composite score.
# error_rate is deliberately NOT averaged in here: a config that drops/errors
# most requests will look artificially *fast* on latency (only the lucky
# survivors get measured), so error rate is a hard gate instead (see
# ERROR_RATE_THRESHOLD), not one-quarter of a linear blend.
SCORING_METRICS = [
    ("ttft_ms_p90", "min"),
    ("request_latency_s_p90", "min"),
    ("tokens_per_second_mean", "max"),
]

# A config that errors out more than this fraction of requests is excluded
# from "best value" ranking entirely, regardless of how fast its survivors look.
ERROR_RATE_THRESHOLD = 0.05

# What to actually look at in the overview/scaling charts.
DISPLAY_METRICS = [
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("ttft_ms_p99", "TTFT p99 (ms)"),
    ("itl_ms_p50", "ITL p50 (ms)"),
    ("itl_ms_p99", "ITL p99 (ms)"),
    ("tokens_per_second_mean", "Throughput, tokens/s (mean)"),
    ("error_rate", "Error rate"),
]

# Per-parameter detail plots use a 3-column x 4-row layout: within each test
# type (rate/concurrent), p50 metrics go in one 3-wide row and p99 below it.
P50_DETAIL_METRICS = [
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("itl_ms_p50", "ITL p50 (ms)"),
    ("tokens_per_second_mean", "Throughput, tokens/s (mean)"),
]
P99_DETAIL_METRICS = [
    ("ttft_ms_p99", "TTFT p99 (ms)"),
    ("itl_ms_p99", "ITL p99 (ms)"),
    ("error_rate", "Error rate"),
]

CMAP = plt.get_cmap("tab10")


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
def prepare_frames(results_dir, manifest_path):
    df = load_results(results_dir=results_dir, manifest_path=manifest_path)
    dropped = sorted(set(df["tag"]) & DROP_TAGS)
    if dropped:
        print("Dropping {} stray/untagged run(s) with no manifest entry: {}".format(
            len(dropped), dropped))
        df = df[~df["tag"].isin(DROP_TAGS)].copy()
    rate_df = df[df["test_type"] == "rate"].copy()
    conc_df = df[df["test_type"] == "concurrent"].copy()
    print("  -> {} rate-test tag(s), {} concurrent-test tag(s).".format(
        rate_df["tag"].nunique(), conc_df["tag"].nunique()))
    return df, rate_df, conc_df


# ---------------------------------------------------------------------------
# Overview: every tag/run compared head-to-head
# ---------------------------------------------------------------------------
def plot_overview_rate(rate_df, out_path):
    if rate_df.empty:
        print("skip overview_rate: no rate-test rows")
        return
    tags = list(rate_df.sort_values("ttft_ms_p50")["tag"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (col, label) in zip(axes.flat, DISPLAY_METRICS):
        vals = [rate_df.set_index("tag").loc[t, col] for t in tags]
        colors = ["#d62728" if t == "baseline" else CMAP(i % 10) for i, t in enumerate(tags)]
        ax.barh(tags, vals, color=colors)
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Rate test (steady offered load) -- all tags compared", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_overview_concurrent(conc_df, out_path):
    if conc_df.empty:
        print("skip overview_concurrent: no concurrent-test rows")
        return
    tags = sorted(conc_df["tag"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (col, label) in zip(axes.flat, DISPLAY_METRICS):
        for i, tag in enumerate(tags):
            sub = conc_df[conc_df["tag"] == tag].sort_values("target")
            style = dict(
                color="#d62728" if tag == "baseline" else CMAP(i % 10),
                linewidth=2.5 if tag == "baseline" else 1.6,
                marker="o",
            )
            ax.plot(sub["target"], sub[col], label=tag, **style)
        ax.set_xlabel("concurrent users")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Concurrent-users test -- all tags compared", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


# ---------------------------------------------------------------------------
# Latency vs. throughput frontier
# ---------------------------------------------------------------------------
def plot_frontier(conc_df, out_path):
    if conc_df.empty:
        print("skip frontier: no concurrent-test rows")
        return
    tags = sorted(conc_df["tag"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, ttft_col, ttft_label in (
        (axes[0], "ttft_ms_p50", "TTFT p50 (ms)"),
        (axes[1], "ttft_ms_p99", "TTFT p99 (ms)"),
    ):
        for i, tag in enumerate(tags):
            sub = conc_df[conc_df["tag"] == tag].sort_values("target")
            style = dict(
                color="#d62728" if tag == "baseline" else CMAP(i % 10),
                linewidth=2.5 if tag == "baseline" else 1.4,
            )
            ax.plot(sub["tokens_per_second_mean"], sub[ttft_col], marker="o", label=tag, **style)
            for _, row in sub.iterrows():
                ax.annotate(
                    str(int(row["target"])),
                    (row["tokens_per_second_mean"], row[ttft_col]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4),
                )
        ax.set_xlabel("Throughput, tokens/s (mean)")
        ax.set_ylabel(ttft_label)
        ax.set_title(ttft_label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Latency/throughput frontier (annotated with concurrency level)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


# ---------------------------------------------------------------------------
# Per-parameter detail views
# ---------------------------------------------------------------------------
def param_rows(df, sweep_item):
    """All rows for one swept parameter, plus synthetic baseline point(s)
    (the 'baseline' tag rows, relabelled to sit at this param's baseline
    value) since baseline is the x=baseline_value point for every parameter's
    comparison. Custom-suffixed baseline runs stay distinct via value_display."""
    key = sweep_item["key"]
    baseline_display = display_value(key, BASELINE[key])
    rows = df[df["param"] == sweep_item["param"]].copy()
    rows["is_baseline"] = False
    base = df[df["param"] == "baseline"].copy()
    base["value"] = BASELINE[key]
    base["value_display"] = base["suffix"].apply(
        lambda s: "{} [{}]".format(baseline_display, s) if s else baseline_display
    )
    base["is_baseline"] = True
    out = pd.concat([base, rows], ignore_index=True)
    return out.drop_duplicates(subset=["value_display", "test_type", "target"])


def plot_param_detail(sweep_item, rate_df, conc_df, out_dir):
    param = sweep_item["param"]
    rate_rows = param_rows(rate_df, sweep_item)
    conc_rows = param_rows(conc_df, sweep_item)
    if len(rate_rows) <= 1 and len(conc_rows) <= 1:
        print("skip {}: no non-baseline data in results/ yet".format(param))
        return None

    fig, axes = plt.subplots(4, 3, figsize=(14, 16))

    def sort_key(v):
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    order = sorted(rate_rows["value_display"].unique(), key=sort_key)
    baseline_labels = set(rate_rows[rate_rows.is_baseline]["value_display"].values)

    def plot_bar_row(axes_row, metrics):
        for ax, (col, label) in zip(axes_row, metrics):
            sub = rate_rows.set_index("value_display").reindex(order)
            colors = ["#d62728" if v in baseline_labels else CMAP(i % 10)
                      for i, v in enumerate(order)]
            ax.bar([str(v) for v in order], sub[col], color=colors)
            ax.set_title(label)
            ax.set_xlabel(param)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.3)

    def plot_line_row(axes_row, metrics, show_legend=False):
        for ax, (col, label) in zip(axes_row, metrics):
            for i, v in enumerate(order):
                sub = conc_rows[conc_rows["value_display"] == v].sort_values("target")
                if sub.empty:
                    continue
                is_base = sub["is_baseline"].iloc[0]
                style = dict(color="#d62728" if is_base else CMAP(i % 10),
                             linewidth=2.5 if is_base else 1.6, marker="o")
                ax.plot(sub["target"], sub[col], label=str(v), **style)
            ax.set_xlabel("concurrent users")
            ax.set_title(label)
            ax.grid(alpha=0.3)
        if show_legend:
            axes_row[0].legend(title=param, fontsize=8, loc="best")

    plot_bar_row(axes[0], P50_DETAIL_METRICS)
    plot_bar_row(axes[1], P99_DETAIL_METRICS)
    plot_line_row(axes[2], P50_DETAIL_METRICS, show_legend=True)
    plot_line_row(axes[3], P99_DETAIL_METRICS)

    fig.suptitle(
        "{} -- rows 1-2: rate test (p50/p99) | rows 3-4: concurrency scaling (p50/p99)".format(param),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.05, 0, 1, 0.96))
    top_pos = axes[0, 0].get_position()
    bottom_pos = axes[2, 0].get_position()
    fig.text(0.01, top_pos.y1, "rate test", fontsize=10, fontweight="bold", va="top")
    fig.text(0.01, bottom_pos.y1, "concurrency\nscaling", fontsize=10, fontweight="bold", va="top")
    out_path = out_dir / "param_{}.png".format(sanitize(param))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)
    return rate_rows


# ---------------------------------------------------------------------------
# Rank each parameter's tested values
# ---------------------------------------------------------------------------
def score_rows(rows, error_threshold=ERROR_RATE_THRESHOLD):
    scored = rows.copy()
    scored["reliable"] = scored["error_rate"] <= error_threshold
    for col, direction in SCORING_METRICS:
        vals = scored[col].astype(float)
        lo, hi = vals.min(), vals.max()
        if hi == lo or pd.isna(hi - lo):
            norm = pd.Series(1.0, index=vals.index)
        else:
            norm = (vals - lo) / (hi - lo)
            if direction == "min":
                norm = 1 - norm
        scored[col + "__score"] = norm
    scored["composite_score"] = scored[[c + "__score" for c, _ in SCORING_METRICS]].mean(axis=1)
    # unreliable rows sort last no matter how good their raw numbers look
    scored.loc[~scored["reliable"], "composite_score"] = -1.0
    return scored.sort_values("composite_score", ascending=False)


def rank_param(sweep_item, rate_rows):
    """Prints the leaderboard for one param and returns (leaderboard_frame,
    recommended_raw_value_or_None)."""
    param = sweep_item["param"]
    key = sweep_item["key"]
    ranked = score_rows(rate_rows).assign(param=param)
    frame = ranked[["param", "value_display", "is_baseline", "reliable", "ttft_ms_p90",
                    "request_latency_s_p90", "tokens_per_second_mean", "error_rate",
                    "composite_score"]]

    print("\n{} (higher composite_score = better; error_rate > {:.0%} excluded, see 'reliable'):".format(
        param, ERROR_RATE_THRESHOLD))
    print(
        ranked[["value_display", "reliable", "ttft_ms_p90", "request_latency_s_p90",
                "tokens_per_second_mean", "error_rate", "composite_score"]]
        .rename(columns={"value_display": "value"})
        .to_string(index=False)
    )

    base_mask = ranked["is_baseline"]
    if base_mask.any():
        base_err = ranked.loc[base_mask, "error_rate"].iloc[0]
        if base_err is not None and base_err > ERROR_RATE_THRESHOLD:
            print("  !! baseline has a {:.0%} error rate here -- its 'fast' latency numbers "
                  "are survivor-biased; treat them with suspicion.".format(base_err))

    recommended = None
    if not ranked["reliable"].any():
        print("  -> NO reliable value found (every value exceeds the error-rate threshold) "
              "-- keeping baseline; investigate this param manually.")
    else:
        best = ranked[ranked["reliable"]].iloc[0]
        if bool(best["is_baseline"]):
            print("  -> best reliable value = baseline ({}); no change recommended.".format(
                display_value(key, BASELINE[key])))
        else:
            recommended = best["value"]  # raw override value
            print("  -> recommend {} = {} (was {})".format(
                key, display_value(key, recommended), display_value(key, BASELINE[key])))
    return frame, recommended


# ---------------------------------------------------------------------------
# Recommended config
# ---------------------------------------------------------------------------
def write_recommended_config(overrides, out_path):
    with open(BASE_PARAMS_FILE) as f:
        manifest = yaml.safe_load(f)
    manifest = copy.deepcopy(manifest)
    manifest["data"].update(overrides)
    manifest.setdefault("metadata", {}).setdefault("labels", {})["sweep-tag"] = "final-rerun"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_param(requested):
    """Map a user-supplied name (param, sanitized param, or substring) to a
    SWEEP item."""
    names = [s["param"] for s in SWEEP]
    for s in SWEEP:
        if requested in (s["param"], sanitize(s["param"])):
            return s
    matches = [s for s in SWEEP if requested in s["param"] or requested in sanitize(s["param"])]
    if len(matches) == 1:
        print("note: '{}' matched param '{}'".format(requested, matches[0]["param"]))
        return matches[0]
    if len(matches) > 1:
        sys.exit("'{}' matches multiple params, be more specific: {}".format(
            requested, [m["param"] for m in matches]))
    sys.exit("param '{}' not found. Available: {}".format(requested, names))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--param", default=None,
        help="plot only this one parameter's sweep (param name, sanitized name, "
        "or unique substring). Skips the overview/frontier/recommendation.")
    ap.add_argument("--results-dir", default=str(REPO / "results" / "outputs"))
    ap.add_argument("--manifest", default=str(HERE / "manifest.jsonl"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument(
        "--no-recommend", action="store_true",
        help="skip writing config/generated/vllm-params.recommended.yaml")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, rate_df, conc_df = prepare_frames(args.results_dir, args.manifest)

    # ---- single-parameter mode -------------------------------------------
    if args.param:
        sweep_item = resolve_param(args.param)
        rate_rows = plot_param_detail(sweep_item, rate_df, conc_df, out_dir)
        if rate_rows is None or len(rate_rows) <= 1:
            print("\nNo usable non-baseline data for '{}' yet.".format(sweep_item["param"]))
            return
        print("\n" + "=" * 78)
        print("LEADERBOARD -- {} (rate test)".format(sweep_item["param"]))
        print("=" * 78)
        rank_param(sweep_item, rate_rows)
        print("\nPlot written to {}".format(out_dir))
        return

    # ---- full sweep mode --------------------------------------------------
    plot_overview_rate(rate_df, out_dir / "00_overview_rate.png")
    plot_overview_concurrent(conc_df, out_dir / "00_overview_concurrent.png")
    plot_frontier(conc_df, out_dir / "01_latency_throughput_frontier.png")

    param_rate_data = {}
    for sweep_item in SWEEP:
        result = plot_param_detail(sweep_item, rate_df, conc_df, out_dir)
        if result is not None:
            param_rate_data[sweep_item["param"]] = result

    print("\n" + "=" * 78)
    print("PER-PARAMETER LEADERBOARD (rate test)")
    print("=" * 78)
    leaderboard_frames = []
    recommended_overrides = {}
    untested = []
    for sweep_item in SWEEP:
        rows = param_rate_data.get(sweep_item["param"])
        if rows is None or len(rows) <= 1:
            untested.append(sweep_item)
            print("\n{}: NO USABLE DATA (baseline only) -- keeping baseline value".format(
                sweep_item["param"]))
            continue
        frame, recommended = rank_param(sweep_item, rows)
        leaderboard_frames.append(frame)
        if recommended is not None:
            recommended_overrides[sweep_item["key"]] = recommended

    leaderboard = pd.concat(leaderboard_frames, ignore_index=True) if leaderboard_frames else pd.DataFrame()
    leaderboard_path = out_dir / "param_rankings.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print("\nwrote", leaderboard_path)

    if untested:
        print("\n{} parameter(s) have no usable measurements yet: {}".format(
            len(untested), [p["param"] for p in untested]))
        for p in untested:
            print("  python3 sweep/run_sweep.py --only {} --resume".format(sanitize(p["param"])))

    if args.no_recommend:
        print("\n(--no-recommend: skipped writing recommended config)")
    else:
        write_recommended_config(recommended_overrides, RECOMMENDED_PARAMS_FILE)
        print("\n" + "=" * 78)
        print("RECOMMENDED CONFIG FOR A FINAL CONFIRMATION RERUN")
        print("=" * 78)
        print("(OFAT sweep -> best-guess combination, not a proven joint optimum)")
        for sweep_item in SWEEP:
            key = sweep_item["key"]
            if key in recommended_overrides:
                print("  {:<24} {} -> {}  [from {} data]".format(
                    key, display_value(key, BASELINE[key]),
                    display_value(key, recommended_overrides[key]), sweep_item["param"]))
            else:
                reason = "no data yet" if sweep_item in untested else "baseline already best"
                print("  {:<24} {}  [unchanged -- {}]".format(
                    key, display_value(key, BASELINE[key]), reason))
        print("\nWrote {}".format(RECOMMENDED_PARAMS_FILE))
        print("To run it:")
        print("  python3 sweep/run_sweep.py --custom-config {} --tag final-rerun".format(
            RECOMMENDED_PARAMS_FILE))

    print("\nPlots written to {}".format(out_dir))


if __name__ == "__main__":
    main()
