#!/usr/bin/env python3
# %% [markdown]
# # Sweep results: plots + "best config" recommendation
#
# Reads `results/aggregated.csv` (produced by `sweep/aggregate.py`) and:
#
# 1. Renders comparison plots across tags/runs for the metrics that matter
#    (TTFT, end-to-end latency, throughput, error rate) — an "overview"
#    chart with every tag side by side, plus one detail chart per swept
#    parameter (bar chart at the steady 20 req/s rate test, line chart
#    showing how it scales across 8/16/32/64 concurrent users).
# 2. Draws a latency-vs-throughput frontier so you can see the tradeoff
#    curve for every config at once.
# 3. Ranks each parameter's tested values with a composite score and prints
#    a leaderboard, then combines the best independently-tested value per
#    parameter into one "recommended" config for a final confirmation rerun.
#
# This file is plain Python split into `# %%` cells, so it runs top to
# bottom as a script (`python3 sweep/plot_results.py`) *and* opens as a
# notebook in Cursor/VS Code ("Run Cell" / "Run Below") without needing
# Jupyter installed.
#
# IMPORTANT CAVEAT: this sweep is one-factor-at-a-time (OFAT), not a full
# grid search — each parameter was varied alone against the baseline, so we
# never measured combinations. The "recommended" config below is the best
# *independent* value per parameter, which assumes the parameters don't
# interact much. That's a reasonable starting guess, not a proven optimum —
# treat the final rerun as a confirmation step, not a formality.

# %%
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
from grid import BASELINE, SWEEP, AUTO_LABEL, display_value, sanitize  # noqa: E402

CSV_PATH = REPO / "results" / "aggregated.csv"
OUT_DIR = REPO / "results" / "plots"
BASE_PARAMS_FILE = REPO / "config" / "vllm-params.yaml"
RECOMMENDED_PARAMS_FILE = REPO / "config" / "generated" / "vllm-params.recommended.yaml"

# Rows with no manifest match (stray/duplicate runs); dropped from the
# "compare tags" views so they don't skew things. See printed note below.
DROP_TAGS = {"20260716-203022"}

# (column, "min" or "max") - direction of "better" for the composite score.
# error_rate is deliberately NOT averaged in here: a config that drops/errors
# most requests will look artificially *fast* on latency (only the lucky
# survivors get measured), so error rate is treated as a hard gate instead
# (see ERROR_RATE_THRESHOLD below), not one-quarter of a linear blend.
SCORING_METRICS = [
    ("ttft_ms_p90", "min"),
    ("request_latency_s_p90", "min"),
    ("tokens_per_second_mean", "max"),
]

# A config that errors out more than this fraction of requests is excluded
# from "best value" ranking entirely, regardless of how fast its survivors
# look -- see the printed warning when this triggers.
ERROR_RATE_THRESHOLD = 0.05

# What to actually look at in the overview/scaling charts.
DISPLAY_METRICS = [
    ("ttft_ms_p90", "TTFT p90 (ms)"),
    ("request_latency_s_p90", "Request latency p90 (s)"),
    ("tokens_per_second_mean", "Throughput, tokens/s (mean)"),
    ("error_rate", "Error rate"),
]

CMAP = plt.get_cmap("tab10")


# %% [markdown]
# ## Load + clean


# %%
def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    dropped = sorted(set(df["tag"]) & DROP_TAGS)
    if dropped:
        print(
            "Dropping {} stray/untagged run(s) from the comparison views: {}\n"
            "(no matching sweep/manifest.jsonl entry -- looks like a "
            "pre-'baseline'-tag leftover run, not part of the OFAT plan)".format(
                len(dropped), dropped
            )
        )
        df = df[~df["tag"].isin(DROP_TAGS)].copy()
    df["value_display"] = df.apply(
        lambda r: AUTO_LABEL if pd.isna(r["value"]) and r["tag"] == "baseline" else r["value"],
        axis=1,
    )
    return df


df = load_data()
rate_df = df[df["test_type"] == "rate"].copy()
conc_df = df[df["test_type"] == "concurrent"].copy()
print(
    "Loaded {} rows: {} rate-test tags, {} concurrent-test tags.".format(
        len(df), rate_df["tag"].nunique(), conc_df["tag"].nunique()
    )
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Overview: every tag/run compared head-to-head
#
# `rate` = steady 20 req/s offered load (one row per tag). `concurrent` =
# fixed 8/16/32/64 concurrent users (up to 4 rows per tag) -- shown as a
# scaling curve instead of a bar since there's a level per point.


# %%
def plot_overview_rate(rate_df, out_path):
    tags = list(rate_df.sort_values("ttft_ms_p90")["tag"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, label) in zip(axes.flat, DISPLAY_METRICS):
        vals = [rate_df.set_index("tag").loc[t, col] for t in tags]
        colors = [
            "#d62728" if t == "baseline" else CMAP(i % 10) for i, t in enumerate(tags)
        ]
        ax.barh(tags, vals, color=colors)
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Rate test (20 req/s offered load) -- all tags compared", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_overview_concurrent(conc_df, out_path):
    tags = sorted(conc_df["tag"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
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


plot_overview_rate(rate_df, OUT_DIR / "00_overview_rate.png")
plot_overview_concurrent(conc_df, OUT_DIR / "00_overview_concurrent.png")


# %% [markdown]
# ## Latency vs. throughput frontier
#
# Every point is one (tag, concurrency-level) run. Down-and-right is better
# (more throughput for less TTFT). Lines connect the same tag across
# concurrency levels so you can see how each config scales.


# %%
def plot_frontier(conc_df, out_path):
    tags = sorted(conc_df["tag"].unique())
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, tag in enumerate(tags):
        sub = conc_df[conc_df["tag"] == tag].sort_values("target")
        style = dict(
            color="#d62728" if tag == "baseline" else CMAP(i % 10),
            linewidth=2.5 if tag == "baseline" else 1.4,
        )
        ax.plot(sub["tokens_per_second_mean"], sub["ttft_ms_p90"], marker="o", label=tag, **style)
        for _, row in sub.iterrows():
            ax.annotate(
                str(int(row["target"])),
                (row["tokens_per_second_mean"], row["ttft_ms_p90"]),
                fontsize=7,
                textcoords="offset points",
                xytext=(4, 4),
            )
    ax.set_xlabel("Throughput, tokens/s (mean)")
    ax.set_ylabel("TTFT p90 (ms)")
    ax.set_title("Latency/throughput frontier (annotated with concurrency level)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


plot_frontier(conc_df, OUT_DIR / "01_latency_throughput_frontier.png")


# %% [markdown]
# ## Per-parameter detail views
#
# For each swept parameter: a bar chart at the rate test (one bar per
# tested value, baseline's own value included in red) and a line chart
# showing how that same set of values scales across concurrency.


# %%
def param_rows(df, sweep_item):
    """All rows for one swept parameter, plus a synthetic baseline point
    (the 'baseline' tag's row, relabeled with this param's baseline value)
    since baseline only appears once in the CSV but is the x=baseline_value
    point for every parameter's comparison."""
    key = sweep_item["key"]
    baseline_value = display_value(key, BASELINE[key])
    rows = df[df["param"] == sweep_item["param"]].copy()
    baseline_rows = df[df["tag"] == "baseline"].copy()
    baseline_rows["value_display"] = baseline_value
    baseline_rows["param"] = sweep_item["param"]
    out = pd.concat([baseline_rows, rows], ignore_index=True)
    out["is_baseline"] = out["value_display"] == baseline_value
    return out.drop_duplicates(subset=["value_display", "test_type", "target"])


def plot_param_detail(df, sweep_item, out_dir):
    param = sweep_item["param"]
    rate_rows = param_rows(rate_df, sweep_item)
    conc_rows = param_rows(conc_df, sweep_item)
    if len(rate_rows) <= 1 and len(conc_rows) <= 1:
        print("skip {}: no non-baseline data in results/ yet".format(param))
        return None

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))

    def sort_key(v):
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    order = sorted(rate_rows["value_display"].unique(), key=sort_key)
    for ax, (col, label) in zip(axes[0], DISPLAY_METRICS):
        sub = rate_rows.set_index("value_display").reindex(order)
        colors = ["#d62728" if v in rate_rows[rate_rows.is_baseline]["value_display"].values
                  else CMAP(i % 10) for i, v in enumerate(order)]
        ax.bar([str(v) for v in order], sub[col], color=colors)
        ax.set_title(label)
        ax.set_xlabel(param)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    for ax, (col, label) in zip(axes[1], DISPLAY_METRICS):
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
    axes[1, 0].legend(title=param, fontsize=8, loc="best")
    fig.suptitle(
        "{} -- top row: rate test (20 req/s) | bottom row: concurrency scaling".format(param),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / "param_{}.png".format(sanitize(param))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)
    return rate_rows


param_rate_data = {}
for sweep_item in SWEEP:
    result = plot_param_detail(df, sweep_item, OUT_DIR)
    if result is not None:
        param_rate_data[sweep_item["param"]] = result


# %% [markdown]
# ## Rank each parameter's tested values + recommend a final config


# %%
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
    # unreliable rows are sorted last no matter how good their raw numbers
    # look (their latency stats are survivor-biased, not comparable)
    scored.loc[~scored["reliable"], "composite_score"] = -1.0
    return scored.sort_values("composite_score", ascending=False)


leaderboard_frames = []
recommended_overrides = {}
untested_params = []

print("\n" + "=" * 78)
print("PER-PARAMETER LEADERBOARD (rate test, 20 req/s offered load)")
print("=" * 78)
for sweep_item in SWEEP:
    param = sweep_item["param"]
    key = sweep_item["key"]
    rows = param_rate_data.get(param)
    if rows is None or len(rows) <= 1:
        untested_params.append(sweep_item)
        print("\n{}: NO USABLE DATA (baseline only) -- keeping baseline value".format(param))
        continue
    ranked = score_rows(rows)
    ranked = ranked.assign(param=param)
    leaderboard_frames.append(
        ranked[["param", "value_display", "is_baseline", "reliable", "ttft_ms_p90",
                "request_latency_s_p90", "tokens_per_second_mean", "error_rate",
                "composite_score"]]
    )
    tested_values = set(rows["value_display"])
    all_values = {display_value(key, v) for v in sweep_item["values"]} | {
        display_value(key, BASELINE[key])
    }
    missing = sorted(all_values - tested_values)
    print("\n{} (higher composite_score = better; error_rate > {:.0%} is excluded, see 'reliable'):".format(
        param, ERROR_RATE_THRESHOLD))
    print(
        ranked[["value_display", "reliable", "ttft_ms_p90", "request_latency_s_p90",
                "tokens_per_second_mean", "error_rate", "composite_score"]]
        .rename(columns={"value_display": "value"})
        .to_string(index=False)
    )
    if missing:
        print("  (not yet measured: {} -- results/*.json missing/corrupt for these)".format(missing))
    base_err = ranked.loc[ranked["is_baseline"], "error_rate"].iloc[0]
    if base_err > ERROR_RATE_THRESHOLD:
        print("  !! baseline ({}) has a {:.0%} error rate at this rate test -- its 'fast' "
              "latency numbers are survivor-biased (only the requests that didn't "
              "error/timeout get measured). Treat baseline's own latency stats here "
              "with suspicion.".format(display_value(key, BASELINE[key]), base_err))

    if not ranked["reliable"].any():
        print("  -> NO reliable value found (every tested value exceeds the error-rate "
              "threshold) -- keeping baseline, but investigate this param/test manually "
              "before trusting any of these numbers.")
    else:
        best = ranked[ranked["reliable"]].iloc[0]
        best_display = best["value_display"]
        if bool(best["is_baseline"]):
            print("  -> best reliable value = baseline ({}); no change recommended.".format(best_display))
        else:
            # map the display value back to the raw override value for this key
            raw_value = next(
                v for v in sweep_item["values"] if display_value(key, v) == best_display
            )
            recommended_overrides[key] = raw_value
            print("  -> recommend {} = {} (was {}, error_rate {:.2%} -> {:.2%})".format(
                key, best_display, display_value(key, BASELINE[key]),
                ranked.loc[ranked["is_baseline"], "error_rate"].iloc[0], best["error_rate"]))

leaderboard = pd.concat(leaderboard_frames, ignore_index=True) if leaderboard_frames else pd.DataFrame()
leaderboard_path = OUT_DIR / "param_rankings.csv"
leaderboard.to_csv(leaderboard_path, index=False)
print("\nwrote", leaderboard_path)

if untested_params:
    print(
        "\n{} parameter(s) have no usable measurements in results/ (files missing or "
        "failed to parse -- see sweep/manifest.jsonl vs. results/*.json): {}".format(
            len(untested_params), [p["param"] for p in untested_params]
        )
    )
    print("Re-run those before trusting a final config, e.g.:")
    for p in untested_params:
        print("  python3 sweep/run_sweep.py --only {} --resume".format(sanitize(p["param"])))


# %% [markdown]
# ## Write the recommended config for a final rerun
#
# Starts from the real `config/vllm-params.yaml` (so model name / EXTRA_ARGS
# etc. are preserved) and overrides only the parameters we have confident,
# independently-tested winners for. Untested parameters are left at baseline.


# %%
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


write_recommended_config(recommended_overrides, RECOMMENDED_PARAMS_FILE)

print("\n" + "=" * 78)
print("RECOMMENDED CONFIG FOR A FINAL CONFIRMATION RERUN")
print("=" * 78)
print("(baseline value shown for keys left unchanged; OFAT sweep -> this is a")
print(" best-guess combination, not a proven joint optimum -- verify with a rerun)")
for sweep_item in SWEEP:
    key = sweep_item["key"]
    if key in recommended_overrides:
        print("  {:<24} {} -> {}  [from {} data]".format(
            key, display_value(key, BASELINE[key]),
            display_value(key, recommended_overrides[key]), sweep_item["param"]))
    else:
        print("  {:<24} {}  [unchanged -- {}]".format(
            key, display_value(key, BASELINE[key]),
            "no data yet" if sweep_item in untested_params else "baseline already best"))
print("\nWrote {}".format(RECOMMENDED_PARAMS_FILE))
print("To run it:")
print("  PARAMS={} ./run.sh all both final-rerun".format(RECOMMENDED_PARAMS_FILE))
print("  ./run.sh pull final-rerun && python3 sweep/aggregate.py && python3 sweep/plot_results.py")

print("\nPlots written to {}".format(OUT_DIR))
