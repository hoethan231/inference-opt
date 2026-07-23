#!/usr/bin/env python3
"""
compare_tags.py - head-to-head comparison of two sweep tags.

`plot_results.py` renders every tag at once, which is great for scanning the
whole OFAT sweep but noisy when you just want to know "how does config A
stack up against config B?" (e.g. baseline vs. a specific tuned value, or two
candidate configs you're deciding between). This script answers exactly that,
for whichever two tags you name.

Reads the pulled GuideLLM result JSONs directly (via load_results.py) and
writes two PNGs to `results/plots/compare/`:

  - `<tag_a>_vs_<tag_b>_rate.png`        - grouped bar chart, rate test
    (single steady-rate offered load -> one bar pair per metric).
  - `<tag_a>_vs_<tag_b>_concurrent.png`  - line chart, concurrent-users test
    (one line per tag, x-axis = concurrency level 8/16/32/64).

Metrics shown (the "important few"): TTFT p50/p99, ITL p50/p99, end-to-end
request latency p50/p99, mean throughput (tokens/s), and error rate.

Also prints a plain-text delta table to stdout (tag_b vs. tag_a, % change,
colored good/bad by which direction is better for that metric) so you get a
quick read without even opening the PNGs.

Usage:
    python3 sweep/compare_tags.py baseline max-num-seqs-256
    python3 sweep/compare_tags.py baseline final-rerun --label-a "current prod" --label-b "recommended"
    python3 sweep/compare_tags.py kv-cache-dtype-fp8-e4m3 kv-cache-dtype-auto --out-dir /tmp/cmp
"""
import argparse
import difflib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; plots are saved as PNG files below
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from load_results import load_results  # noqa: E402

RESULTS_DIR = REPO / "results" / "outputs"
MANIFEST_PATH = HERE / "manifest.jsonl"
OUT_DIR = REPO / "results" / "plots" / "compare"

CMAP = plt.get_cmap("tab10")
COLOR_A = CMAP(0)  # blue
COLOR_B = CMAP(1)  # orange
GOOD_COLOR = "#2ca02c"
BAD_COLOR = "#d62728"
NEUTRAL_COLOR = "#7f7f7f"

# (column, display label, "min" or "max" = direction of "better")
METRICS = [
    ("ttft_ms_p50", "TTFT p50 (ms)", "min"),
    ("ttft_ms_p99", "TTFT p99 (ms)", "min"),
    ("itl_ms_p50", "ITL p50 (ms)", "min"),
    ("itl_ms_p99", "ITL p99 (ms)", "min"),
    ("request_latency_s_p50", "E2E latency p50 (s)", "min"),
    ("request_latency_s_p99", "E2E latency p99 (s)", "min"),
    ("tokens_per_second_mean", "Throughput, tokens/s (mean)", "max"),
    ("error_rate", "Error rate", "min"),
]


# %% [markdown]
# ## Load + resolve tags


# %%
def resolve_tag(df, requested):
    """Exact match first; falls back to a unique substring match; otherwise
    exits with the closest suggestions (typos / partial tags are common when
    typing these by hand)."""
    tags = df["tag"].unique().tolist()
    if requested in tags:
        return requested
    substring_matches = [t for t in tags if requested in t]
    if len(substring_matches) == 1:
        print("note: '{}' matched tag '{}' by substring".format(requested, substring_matches[0]))
        return substring_matches[0]
    if len(substring_matches) > 1:
        sys.exit(
            "'{}' matches multiple tags, be more specific: {}".format(
                requested, substring_matches
            )
        )
    suggestions = difflib.get_close_matches(requested, tags, n=5)
    sys.exit(
        "tag '{}' not found in results/.\n{}".format(
            requested,
            "Did you mean: {}".format(suggestions) if suggestions else
            "Available tags: {}".format(sorted(tags)),
        )
    )


# %% [markdown]
# ## Delta table (printed to stdout)


# %%
def fmt_delta(val_a, val_b, direction):
    if pd.isna(val_a) or pd.isna(val_b):
        return "n/a", NEUTRAL_COLOR
    if val_a == 0:
        return ("n/a (base=0)", NEUTRAL_COLOR) if val_b == 0 else ("+inf%", BAD_COLOR)
    pct = (val_b - val_a) / abs(val_a) * 100
    better = (direction == "min" and pct < 0) or (direction == "max" and pct > 0)
    color = GOOD_COLOR if pct != 0 and better else (BAD_COLOR if pct != 0 else NEUTRAL_COLOR)
    sign = "+" if pct > 0 else ""
    return "{}{:.1f}%".format(sign, pct), color


def print_delta_table(rate_a, rate_b, conc_a, conc_b, label_a, label_b):
    print("\n" + "=" * 88)
    print("RATE TEST -- {} vs. {}".format(label_a, label_b))
    print("=" * 88)
    print("{:<28} {:>14} {:>14} {:>14}".format("metric", label_a, label_b, "delta"))
    for col, label, direction in METRICS:
        va = rate_a[col].iloc[0] if not rate_a.empty and col in rate_a else float("nan")
        vb = rate_b[col].iloc[0] if not rate_b.empty and col in rate_b else float("nan")
        delta_str, _ = fmt_delta(va, vb, direction)
        print("{:<28} {:>14} {:>14} {:>14}".format(label, _fmt_val(va), _fmt_val(vb), delta_str))

    if conc_a.empty and conc_b.empty:
        return
    targets = sorted(set(conc_a["target"]) | set(conc_b["target"]))
    print("\n" + "=" * 88)
    print("CONCURRENT-USERS TEST -- {} vs. {}".format(label_a, label_b))
    print("=" * 88)
    for target in targets:
        ta = conc_a[conc_a["target"] == target]
        tb = conc_b[conc_b["target"] == target]
        print("\n-- concurrency = {} --".format(int(target) if float(target).is_integer() else target))
        print("{:<28} {:>14} {:>14} {:>14}".format("metric", label_a, label_b, "delta"))
        for col, label, direction in METRICS:
            va = ta[col].iloc[0] if not ta.empty and col in ta else float("nan")
            vb = tb[col].iloc[0] if not tb.empty and col in tb else float("nan")
            delta_str, _ = fmt_delta(va, vb, direction)
            print("{:<28} {:>14} {:>14} {:>14}".format(label, _fmt_val(va), _fmt_val(vb), delta_str))


def _fmt_val(v):
    if pd.isna(v):
        return "n/a"
    return "{:.4g}".format(v)


# %% [markdown]
# ## Rate test: grouped bar chart


# %%
def plot_rate_comparison(rate_a, rate_b, label_a, label_b, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (col, label, direction) in zip(axes.flat, METRICS):
        va = rate_a[col].iloc[0] if not rate_a.empty and col in rate_a else float("nan")
        vb = rate_b[col].iloc[0] if not rate_b.empty and col in rate_b else float("nan")
        bars = ax.bar([label_a, label_b], [va, vb], color=[COLOR_A, COLOR_B])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, [va, vb]):
            if not pd.isna(v):
                ax.annotate(
                    "{:.4g}".format(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8,
                )
        delta_str, color = fmt_delta(va, vb, direction)
        ax.annotate(
            "{} vs {}: {}".format(label_b, label_a, delta_str),
            xy=(0.5, 1.12), xycoords="axes fraction", ha="center", fontsize=8,
            color=color, fontweight="bold",
        )
    fig.suptitle(
        "Rate test -- {} vs. {}".format(label_a, label_b), fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


# %% [markdown]
# ## Concurrent-users test: scaling line chart


# %%
def plot_concurrent_comparison(conc_a, conc_b, label_a, label_b, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (col, label, _direction) in zip(axes.flat, METRICS):
        for sub, lbl, color in ((conc_a, label_a, COLOR_A), (conc_b, label_b, COLOR_B)):
            if sub.empty or col not in sub:
                continue
            s = sub.sort_values("target")
            ax.plot(s["target"], s[col], label=lbl, color=color, marker="o", linewidth=2)
        ax.set_xlabel("concurrent users")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Concurrent-users test -- {} vs. {}".format(label_a, label_b), fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


# %% [markdown]
# ## Latency-vs-throughput frontier (bonus: same idea as plot_results.py,
# ## just for these two tags)


# %%
def plot_frontier_comparison(conc_a, conc_b, label_a, label_b, out_path):
    if conc_a.empty and conc_b.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ttft_col, ttft_label in (
        (axes[0], "ttft_ms_p50", "TTFT p50 (ms)"),
        (axes[1], "ttft_ms_p99", "TTFT p99 (ms)"),
    ):
        for sub, lbl, color in ((conc_a, label_a, COLOR_A), (conc_b, label_b, COLOR_B)):
            if sub.empty:
                continue
            s = sub.sort_values("target")
            ax.plot(s["tokens_per_second_mean"], s[ttft_col], marker="o", label=lbl,
                     color=color, linewidth=2)
            for _, row in s.iterrows():
                ax.annotate(
                    str(int(row["target"])), (row["tokens_per_second_mean"], row[ttft_col]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4),
                )
        ax.set_xlabel("Throughput, tokens/s (mean)")
        ax.set_ylabel(ttft_label)
        ax.set_title(ttft_label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(
        "Latency/throughput frontier -- {} vs. {} (annotated with concurrency level)".format(
            label_a, label_b),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


# %% [markdown]
# ## Main


# %%
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("tag_a", help="first tag (e.g. 'baseline')")
    ap.add_argument("tag_b", help="second tag to compare against tag_a")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="dir of pulled result JSONs")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH), help="path to manifest.jsonl")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="directory to write PNGs to")
    ap.add_argument("--label-a", default=None, help="display label for tag_a (default: the tag itself)")
    ap.add_argument("--label-b", default=None, help="display label for tag_b (default: the tag itself)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir=args.results_dir, manifest_path=args.manifest)
    tag_a = resolve_tag(df, args.tag_a)
    tag_b = resolve_tag(df, args.tag_b)
    if tag_a == tag_b:
        sys.exit("tag_a and tag_b resolved to the same tag ('{}') -- nothing to compare.".format(tag_a))
    label_a = args.label_a or tag_a
    label_b = args.label_b or tag_b

    df_a = df[df["tag"] == tag_a]
    df_b = df[df["tag"] == tag_b]
    rate_a = df_a[df_a["test_type"] == "rate"]
    rate_b = df_b[df_b["test_type"] == "rate"]
    conc_a = df_a[df_a["test_type"] == "concurrent"]
    conc_b = df_b[df_b["test_type"] == "concurrent"]

    if rate_a.empty and conc_a.empty:
        sys.exit("no rows found for tag '{}' in results/".format(tag_a))
    if rate_b.empty and conc_b.empty:
        sys.exit("no rows found for tag '{}' in results/".format(tag_b))

    print_delta_table(rate_a, rate_b, conc_a, conc_b, label_a, label_b)

    stem = "{}_vs_{}".format(tag_a, tag_b)
    if not rate_a.empty or not rate_b.empty:
        plot_rate_comparison(rate_a, rate_b, label_a, label_b, out_dir / "{}_rate.png".format(stem))
    else:
        print("\nskipping rate plot: no rate-test rows for either tag")

    if not conc_a.empty or not conc_b.empty:
        plot_concurrent_comparison(conc_a, conc_b, label_a, label_b, out_dir / "{}_concurrent.png".format(stem))
        plot_frontier_comparison(conc_a, conc_b, label_a, label_b, out_dir / "{}_frontier.png".format(stem))
    else:
        print("skipping concurrent/frontier plots: no concurrent-test rows for either tag")

    print("\nPlots written to {}".format(out_dir))


if __name__ == "__main__":
    main()
