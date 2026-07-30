#!/usr/bin/env python3
"""
plot_grid.py - comparison plots + a ranked leaderboard for a grid search.

Consumes the grid results produced by sweep/run_grid.py (the pulled GuideLLM
JSONs in results/outputs/ + sweep/grid_manifest.jsonl) and writes, to
results/plots/grid/, five files prefixed with the experiment name (the 'name'
in the grid config, e.g. 'laguna' -> laguna_frontier.png):

  1. <name>_overview_rate.png        - every combo bar-charted head-to-head on
                                       the steady-rate test (TTFT/ITL/
                                       throughput/error rate).
  2. <name>_overview_concurrent.png  - the same metrics as lines vs. concurrency
                                       (8/16/32/64 users), one line per combo.
  3. <name>_frontier.png             - latency-vs-throughput frontier, one line
                                       per combo, annotated by concurrency level.
  4. <name>_ranking.png              - combos sorted by composite score.
  5. <name>_ranking.csv              - the full ranked leaderboard (one row per
                                       combo, with each swept axis broken out
                                       into its own column).

Because grid_manifest.jsonl accumulates every grid ever run, the name both
filters which combos are plotted and prefixes the output files. Without that,
a new experiment silently overwrites the previous one's plots and blends
unrelated models into one leaderboard.

Scoring reuses plot_results.py exactly: a config that errors out more than
ERROR_RATE_THRESHOLD of requests is gated out ("reliable" = False) so its
survivor-biased latency can't win, and the rest are ranked by a 0..1 composite
of TTFT p90 (min), end-to-end latency p90 (min), and mean throughput (max) on
the rate test.

Unlike the OFAT recommendation in plot_results.py, this leaderboard IS a
direct measurement of joint combinations, so the #1 row is a genuinely
observed best config, not a per-axis best-guess.

Usage:
    python3 sweep/plot_grid.py                      # name from config/grid.json
    python3 sweep/plot_grid.py --name laguna        # only tags 'laguna-*'
    python3 sweep/plot_grid.py --all                # every grid combo, as 'all_*'
    python3 sweep/plot_grid.py --top 10             # annotate/limit ranking
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; plots are saved as PNG files below
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from grid import sanitize  # noqa: E402
from load_results import load_manifest, load_results  # noqa: E402
from plot_results import (  # noqa: E402
    CMAP,
    DISPLAY_METRICS,
    ERROR_RATE_THRESHOLD,
    SCORING_METRICS,
    score_rows,
)

GRID_MANIFEST = HERE / "grid_manifest.jsonl"
OUT_DIR = REPO / "results" / "plots" / "grid"
DEFAULT_GRID_CONFIG = REPO / "config" / "grid.json"
# Deliberately not "grid": an experiment can itself be named 'grid', and --all
# must not overwrite that experiment's own plots.
ALL_PREFIX = "all"


def name_from_config(path):
    """The grid config's 'name', sanitized the same way run_grid.py builds tags
    from it. Returns None if the config is missing/unreadable, so plotting
    still works standalone."""
    try:
        with open(path) as f:
            name = json.load(f).get("name")
    except (OSError, ValueError):
        return None
    return sanitize(name) if name else None


# ---------------------------------------------------------------------------
# Labels: derive a compact, readable name per combo from the swept axes only
# ---------------------------------------------------------------------------
def combo_overrides(manifest, tags):
    """tag -> overrides dict, for the grid tags we're plotting."""
    return {t: (manifest.get(t, {}).get("overrides") or {}) for t in tags}


def varying_keys(overrides_by_tag):
    """Keys whose value differs across combos (the axes actually swept). If
    nothing varies (single combo), fall back to every key present."""
    all_keys = sorted({k for ov in overrides_by_tag.values() for k in ov})
    varying = [k for k in all_keys
               if len({str(ov.get(k)) for ov in overrides_by_tag.values()}) > 1]
    return varying or all_keys


def make_labels(overrides_by_tag, keys):
    """tag -> 'K=v, K=v' using only the swept axes (compact but unambiguous)."""
    labels = {}
    for tag, ov in overrides_by_tag.items():
        parts = ["{}={}".format(k, ov.get(k, "?")) for k in keys]
        labels[tag] = ", ".join(parts) if parts else tag
    return labels


# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------
def prepare(results_dir, manifest_path, name):
    df = load_results(results_dir=results_dir, manifest_path=manifest_path)
    df = df[df["param"] == "grid"].copy()
    if name:
        prefix = "{}-".format(name)
        df = df[df["tag"].str.startswith(prefix)].copy()
    if df.empty:
        sys.exit(
            "no grid results found in {}.\n"
            "Did you run the grid and pull results? e.g.\n"
            "  ./sweep/run_grid.py\n"
            "(filter --name '{}' matched nothing)".format(results_dir, name or "")
        )

    manifest = load_manifest(manifest_path)
    tags = sorted(df["tag"].unique())
    ov_by_tag = combo_overrides(manifest, tags)
    keys = varying_keys(ov_by_tag)
    labels = make_labels(ov_by_tag, keys)
    df["label"] = df["tag"].map(labels)

    rate_df = df[df["test_type"] == "rate"].copy()
    conc_df = df[df["test_type"] == "concurrent"].copy()
    print("Loaded {} grid combo(s): {} rate, {} concurrent. Swept axes: {}".format(
        len(tags), rate_df["tag"].nunique(), conc_df["tag"].nunique(), keys))
    return df, rate_df, conc_df, ov_by_tag, keys, labels


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
def plot_overview_rate(rate_df, out_path, title):
    if rate_df.empty:
        print("skip overview_rate: no rate-test rows")
        return
    ordered = rate_df.sort_values("ttft_ms_p50")
    labels = list(ordered["label"])
    fig, axes = plt.subplots(2, 3, figsize=(20, max(6, 0.5 * len(labels) + 4)))
    for ax, (col, disp) in zip(axes.flat, DISPLAY_METRICS):
        ax.barh(labels, ordered[col], color=[CMAP(i % 10) for i in range(len(labels))])
        ax.invert_yaxis()
        ax.set_xlabel(disp)
        ax.set_title(disp)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("{} -- rate test (steady offered load), every combo compared".format(title),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_overview_concurrent(conc_df, out_path, title):
    if conc_df.empty:
        print("skip overview_concurrent: no concurrent-test rows")
        return
    labels = sorted(conc_df["label"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (col, disp) in zip(axes.flat, DISPLAY_METRICS):
        for i, label in enumerate(labels):
            sub = conc_df[conc_df["label"] == label].sort_values("target")
            ax.plot(sub["target"], sub[col], marker="o", color=CMAP(i % 10),
                    linewidth=1.6, label=label)
        ax.set_xlabel("concurrent users")
        ax.set_ylabel(disp)
        ax.set_title(disp)
        ax.grid(alpha=0.3)
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=min(4, len(labels)),
               fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("{} -- concurrent-users test, every combo compared".format(title),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_frontier(conc_df, out_path, title):
    if conc_df.empty:
        print("skip frontier: no concurrent-test rows")
        return
    labels = sorted(conc_df["label"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, ttft_col, ttft_label in (
        (axes[0], "ttft_ms_p50", "TTFT p50 (ms)"),
        (axes[1], "ttft_ms_p99", "TTFT p99 (ms)"),
    ):
        for i, label in enumerate(labels):
            sub = conc_df[conc_df["label"] == label].sort_values("target")
            ax.plot(sub["tokens_per_second_mean"], sub[ttft_col], marker="o",
                    color=CMAP(i % 10), linewidth=1.5, label=label)
            for _, row in sub.iterrows():
                ax.annotate(str(int(row["target"])),
                            (row["tokens_per_second_mean"], row[ttft_col]),
                            fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("Throughput, tokens/s (mean)")
        ax.set_ylabel(ttft_label)
        ax.set_title(ttft_label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("{} -- latency/throughput frontier (annotated with concurrency)".format(title),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def ranking_frame(rate_df, conc_df):
    """One row per combo to rank. Prefer the rate test (steady load); fall back
    to the per-combo mean over concurrency levels if only concurrent data
    exists."""
    if not rate_df.empty:
        return rate_df, "rate test"
    agg_cols = [c for c, _ in SCORING_METRICS] + ["error_rate"]
    grouped = (conc_df.groupby(["tag", "label"], as_index=False)[agg_cols]
               .mean())
    return grouped, "concurrent test (mean over users)"


def build_leaderboard(rank_df, ov_by_tag, keys):
    scored = score_rows(rank_df).reset_index(drop=True)
    scored.insert(0, "rank", scored.index + 1)
    for key in keys:
        scored[key] = scored["tag"].map(lambda t: ov_by_tag.get(t, {}).get(key, ""))
    cols = (["rank", "label"] + keys + ["reliable"]
            + [c for c, _ in SCORING_METRICS] + ["error_rate", "composite_score", "tag"])
    cols = [c for c in cols if c in scored.columns]
    return scored[cols]


def plot_ranking(leaderboard, out_path, top, title):
    board = leaderboard.head(top) if top else leaderboard
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(board) + 2)))
    colors = ["#2ca02c" if r else "#d62728" for r in board["reliable"]]
    ax.barh(board["label"], board["composite_score"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("composite score (higher = better; red = failed reliability gate)")
    ax.set_title("{} ranking -- best parameter combinations".format(title))
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)
    for i, (_, row) in enumerate(board.iterrows()):
        ax.annotate("{:.3f}".format(row["composite_score"]),
                    (row["composite_score"], i), va="center", fontsize=8,
                    xytext=(3, 0), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO / "results" / "outputs"))
    ap.add_argument("--manifest", default=str(GRID_MANIFEST))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--config", default=str(DEFAULT_GRID_CONFIG),
                    help="grid config to read the experiment 'name' from when "
                    "--name is omitted (default: config/grid.json)")
    ap.add_argument("--name", default=None,
                    help="only plot combos whose tag starts with '<name>-', and "
                    "prefix the output files with it; default: the 'name' in "
                    "--config")
    ap.add_argument("--all", action="store_true",
                    help="plot every grid combo in the manifest regardless of "
                    "name, writing '{}_*' files".format(ALL_PREFIX))
    ap.add_argument("--top", type=int, default=None,
                    help="limit the ranking chart to the top N combos "
                    "(the CSV always contains all of them)")
    args = ap.parse_args()

    if args.all:
        name = None
    else:
        name = args.name or name_from_config(args.config)
        if not name:
            sys.exit(
                "could not determine the experiment name: no --name given and "
                "no 'name' readable from {}.\nPass --name <prefix>, or --all to "
                "plot every grid combo.".format(args.config)
            )
    prefix = name or ALL_PREFIX
    title = "Grid search [{}]".format(name) if name else "Grid search"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, rate_df, conc_df, ov_by_tag, keys, _labels = prepare(
        args.results_dir, args.manifest, name)

    plot_overview_rate(rate_df, out_dir / "{}_overview_rate.png".format(prefix), title)
    plot_overview_concurrent(
        conc_df, out_dir / "{}_overview_concurrent.png".format(prefix), title)
    plot_frontier(conc_df, out_dir / "{}_frontier.png".format(prefix), title)

    rank_df, basis = ranking_frame(rate_df, conc_df)
    leaderboard = build_leaderboard(rank_df, ov_by_tag, keys)

    print("\n" + "=" * 88)
    print("{} LEADERBOARD ({}; error_rate > {:.0%} gated out via 'reliable')".format(
        prefix.upper(), basis, ERROR_RATE_THRESHOLD))
    print("=" * 88)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(leaderboard.drop(columns=["tag"]).to_string(index=False))

    csv_path = out_dir / "{}_ranking.csv".format(prefix)
    leaderboard.to_csv(csv_path, index=False)
    print("\nwrote", csv_path)

    plot_ranking(leaderboard, out_dir / "{}_ranking.png".format(prefix), args.top, title)

    reliable = leaderboard[leaderboard["reliable"]]
    if reliable.empty:
        print("\nNO reliable combo (every combo exceeds the {:.0%} error-rate gate) "
              "-- investigate before trusting any latency numbers.".format(
                  ERROR_RATE_THRESHOLD))
    else:
        best = reliable.iloc[0]
        print("\nBest combo: {}  (composite {:.3f})".format(
            best["label"], best["composite_score"]))
        print("  tag: {}".format(best["tag"]))
        print("  overrides: {}".format(ov_by_tag.get(best["tag"], {})))
        print("\nConfirm it head-to-head against your baseline with:")
        print("  python3 sweep/compare_tags.py baseline {} --manifest {}".format(
            best["tag"], args.manifest))

    print("\nPlots written to {}".format(out_dir))


if __name__ == "__main__":
    main()
