"""Plots for whatever has run.

Replaces plot_results.py (OFAT) and plot_grid.py (grid, which imported half of
plot_results). There is one code path: the shape of the data decides what gets
drawn. A steady-rate workload becomes bar charts, a concurrency ladder becomes
scaling lines plus a latency/throughput frontier, and an experiment with named
axes additionally gets a per-axis detail figure.

Deliberately no ranking, no composite score and no leaderboard: collapsing TTFT,
latency and throughput into one number hides the tradeoff that the plots exist to
show, and which knob "wins" depends on what you are optimising for. The plots and
`bench compare` present the measurements; the judgement stays with the reader.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: everything is written to PNG

import matplotlib.pyplot as plt
import pandas as pd

from . import ingest, settings, util
from .store import RunRecord

CMAP = plt.get_cmap("tab10")
BASELINE_COLOR = "#d62728"

# Above this, a run's latency numbers describe its survivors rather than the load
# it was given, so the report says so explicitly.
ERROR_RATE_NOTABLE = 0.05

# Six panels: p50 on the left of each pair, p99 on the right.
#
# Throughput here is output_tokens_per_second, NOT tokens_per_second. GuideLLM's
# tokens_per_second reports a whole-run aggregate as its mean but windowed
# samples as its percentiles, so those percentiles are wild (p50 36 tok/s, p99
# 405,869 tok/s for the same run). output_tokens_per_second is one consistent
# distribution, which is what a p50/p99 pair needs to mean anything.
OVERVIEW_METRICS = [
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("ttft_ms_p99", "TTFT p99 (ms)"),
    ("itl_ms_p50", "ITL p50 (ms)"),
    ("itl_ms_p99", "ITL p99 (ms)"),
    ("output_tps_p50", "Output throughput p50, tokens/s"),
    ("output_tps_p99", "Output throughput p99, tokens/s"),
]

DETAIL_P50 = [
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("itl_ms_p50", "ITL p50 (ms)"),
    ("output_tps_p50", "Output throughput p50, tokens/s"),
]
DETAIL_P99 = [
    ("ttft_ms_p99", "TTFT p99 (ms)"),
    ("itl_ms_p99", "ITL p99 (ms)"),
    ("output_tps_p99", "Output throughput p99, tokens/s"),
]


def _colors(labels: list[str], baselines: set[str]) -> list[str]:
    return [
        BASELINE_COLOR if label in baselines else CMAP(i % 10)
        for i, label in enumerate(labels)
    ]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    try:
        shown = path.relative_to(settings.REPO)
    except ValueError:
        shown = path
    util.step(f"wrote {shown}")
    return path


def _sort_key(frame: pd.DataFrame):
    """Order series numerically when the values are numbers, else alphabetically."""
    lookup = frame.drop_duplicates("series").set_index("series")["value"].to_dict()

    def key(series: str):
        raw = lookup.get(series, series)
        try:
            return (0, float(raw), series)
        except (TypeError, ValueError):
            return (1, str(raw), series)

    return key


def plot_rate_overview(df: pd.DataFrame, path: Path, title: str) -> None:
    """Every run compared head-to-head at a fixed offered load."""
    ordered = df.sort_values("ttft_ms_p50")
    labels = list(ordered["series"])
    baselines = set(df[df["param"] == "baseline"]["series"])
    height = max(6.0, 0.42 * len(labels) + 4)
    fig, axes = plt.subplots(2, 3, figsize=(20, height))
    for ax, (column, label) in zip(axes.flat, OVERVIEW_METRICS):
        ax.barh(labels, ordered[column], color=_colors(labels, baselines))
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"{title} — steady offered load", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)


def plot_scaling(df: pd.DataFrame, path: Path, title: str) -> None:
    """How each run's latency and throughput move with concurrency."""
    order = sorted(df["series"].unique(), key=_sort_key(df))
    baselines = set(df[df["param"] == "baseline"]["series"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (column, label) in zip(axes.flat, OVERVIEW_METRICS):
        for i, series in enumerate(order):
            subset = df[df["series"] == series].sort_values("target")
            is_base = series in baselines
            ax.plot(
                subset["target"],
                subset[column],
                marker="o",
                label=series,
                color=BASELINE_COLOR if is_base else CMAP(i % 10),
                linewidth=2.5 if is_base else 1.6,
            )
        ax.set_xlabel("concurrent users")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, max(1, len(order))),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(f"{title} — scaling with concurrency", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    _save(fig, path)


def plot_frontier(df: pd.DataFrame, path: Path, title: str) -> None:
    """Latency against throughput: the tradeoff you actually pick a config on."""
    order = sorted(df["series"].unique(), key=_sort_key(df))
    baselines = set(df[df["param"] == "baseline"]["series"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Each panel keeps both axes at the same percentile: a p99 latency read
    # against mean throughput would flatter a config that is erratic under load.
    for ax, column, label, throughput, throughput_label in (
        (
            axes[0],
            "ttft_ms_p50",
            "TTFT p50 (ms)",
            "output_tps_p50",
            "Output throughput p50, tokens/s",
        ),
        (
            axes[1],
            "ttft_ms_p99",
            "TTFT p99 (ms)",
            "output_tps_p99",
            "Output throughput p99, tokens/s",
        ),
    ):
        for i, series in enumerate(order):
            subset = df[df["series"] == series].sort_values("target")
            is_base = series in baselines
            ax.plot(
                subset[throughput],
                subset[column],
                marker="o",
                label=series,
                color=BASELINE_COLOR if is_base else CMAP(i % 10),
                linewidth=2.5 if is_base else 1.5,
            )
            for _, row in subset.iterrows():
                ax.annotate(
                    str(int(row["target"])),
                    (row[throughput], row[column]),
                    fontsize=7,
                    textcoords="offset points",
                    xytext=(4, 4),
                )
        ax.set_xlabel(throughput_label)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title} — latency/throughput frontier (points labelled with concurrency)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def _axis_key(rows: pd.DataFrame, records: dict[str, RunRecord]) -> str | None:
    """The ConfigMap key an axis varied, recovered from any of its runs."""
    for tag in rows["tag"].unique():
        overrides = (records.get(tag).overrides if tag in records else {}) or {}
        if len(overrides) == 1:
            return next(iter(overrides))
    return None


def _with_baseline(
    df: pd.DataFrame, param: str, records: dict[str, RunRecord]
) -> pd.DataFrame:
    """One axis's rows, plus the baseline run shown at its own value.

    The baseline is the x = "unchanged" point of every axis, so it belongs on
    every per-axis chart. Its value for this axis is read from the config the
    baseline actually ran with, rather than a duplicated table of defaults.
    """
    rows = df[df["param"] == param].copy()
    if rows.empty:
        return rows
    rows["is_baseline"] = False

    baseline = df[df["param"] == "baseline"].copy()
    if baseline.empty:
        return rows

    key = _axis_key(rows, records)
    baseline["is_baseline"] = True
    if key:
        def baseline_value(tag: str) -> str:
            record = records.get(tag)
            return (record.params or {}).get(key, "baseline") if record else "baseline"

        baseline["value"] = baseline["tag"].map(baseline_value)
        baseline["value_display"] = baseline.apply(
            lambda r: f"{r['value']} [{r['suffix']}]" if r["suffix"] else str(r["value"]),
            axis=1,
        )
        if df["model"].nunique() > 1:
            baseline["series"] = baseline["model"] + ": " + baseline["value_display"]
        else:
            baseline["series"] = baseline["value_display"]

    combined = pd.concat([baseline, rows], ignore_index=True)
    return combined.drop_duplicates(subset=["series", "workload", "target"])


def plot_axis_detail(
    param: str, rate: pd.DataFrame, conc: pd.DataFrame, path: Path
) -> pd.DataFrame | None:
    """Bar charts at steady load over scaling lines, p50 above p99."""
    if rate.empty and conc.empty:
        return None
    reference = rate if not rate.empty else conc
    order = sorted(reference["series"].unique(), key=_sort_key(reference))
    baselines = (
        set(reference[reference["is_baseline"]]["series"])
        if "is_baseline" in reference
        else set()
    )

    rows = 4 if not conc.empty else 2
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    axes = axes.reshape(rows, 3)

    def bars(axes_row, metrics):
        indexed = rate.drop_duplicates("series").set_index("series").reindex(order)
        for ax, (column, label) in zip(axes_row, metrics):
            ax.bar(order, indexed[column], color=_colors(order, baselines))
            ax.set_title(label)
            ax.set_xlabel(param)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.grid(axis="y", alpha=0.3)

    def lines(axes_row, metrics, legend=False):
        for ax, (column, label) in zip(axes_row, metrics):
            for i, series in enumerate(order):
                subset = conc[conc["series"] == series].sort_values("target")
                if subset.empty:
                    continue
                is_base = series in baselines
                ax.plot(
                    subset["target"],
                    subset[column],
                    marker="o",
                    label=series,
                    color=BASELINE_COLOR if is_base else CMAP(i % 10),
                    linewidth=2.5 if is_base else 1.6,
                )
            ax.set_xlabel("concurrent users")
            ax.set_title(label)
            ax.grid(alpha=0.3)
        if legend:
            axes_row[0].legend(title=param, fontsize=8, loc="best")

    if not rate.empty:
        bars(axes[0], DETAIL_P50)
        bars(axes[1], DETAIL_P99)
        if not conc.empty:
            lines(axes[2], DETAIL_P50, legend=True)
            lines(axes[3], DETAIL_P99)
    else:
        lines(axes[0], DETAIL_P50, legend=True)
        lines(axes[1], DETAIL_P99)

    subtitle = (
        "rows 1-2: steady load (p50/p99) | rows 3-4: concurrency scaling (p50/p99)"
        if not rate.empty and not conc.empty
        else "concurrency scaling (p50/p99)"
    )
    fig.suptitle(f"{param} — {subtitle}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
    return rate if not rate.empty else conc


def warn_unusable(df: pd.DataFrame) -> None:
    """Name the runs whose numbers can't be taken at face value.

    Nothing ranks or scores runs any more, and error rate is no longer plotted,
    so this is the one place a broken run announces itself. Both cases below read
    as *good* on a latency chart, which is exactly why they need saying:
    a run that served nothing has no latency, and a run that failed most requests
    only reports its lucky survivors.
    """
    per_run = df.drop_duplicates(subset=["tag", "workload"])

    empty = per_run[pd.to_numeric(per_run["requests_successful"], errors="coerce").fillna(0) <= 0]
    if not empty.empty:
        names = sorted(set(empty["series"].astype(str)))
        util.warn(
            f"{len(names)} run(s) completed no requests at all, so their lines sit "
            f"at zero rather than fast: {', '.join(names[:6])}"
        )

    errors = pd.to_numeric(per_run["error_rate"], errors="coerce")
    noisy = per_run[errors.fillna(0) > ERROR_RATE_NOTABLE]
    noisy = noisy[~noisy["tag"].isin(empty["tag"])]
    if not noisy.empty:
        rows = [
            [str(row["series"]), row["workload"], util.fmt(row["error_rate"], 3)]
            for _, row in noisy.sort_values("error_rate", ascending=False).iterrows()
        ]
        print()
        util.rule(f"runs with an error rate above {ERROR_RATE_NOTABLE:.0%}")
        print(util.table(["run", "workload", "error rate"], rows))
        util.step("their latency numbers describe the requests that survived, not the load")


def build(
    records: list[RunRecord],
    out_dir: Path,
    title: str,
    *,
    param: str | None = None,
) -> None:
    by_tag = {r.tag: r for r in records}
    df = ingest.frame(records)
    out_dir.mkdir(parents=True, exist_ok=True)

    # "unknown" is what an import gets when no axis was ever recorded for it: a
    # bag of unrelated one-offs, which is not an axis and can't be plotted as one.
    axis_params = [
        p
        for p in df["param"].unique()
        if p not in ("baseline", "single", "grid", "unknown", "")
    ]

    if param:
        matches = [p for p in axis_params if param in p or param in util.sanitize(p)]
        if not matches:
            util.die(
                f"no axis named '{param}' in these results. Present: "
                f"{', '.join(axis_params) or '(none)'}"
            )
        axis_params = matches[:1]
        util.log(f"scoped to axis '{axis_params[0]}'")
    else:
        # Overview + frontier across everything that ran.
        for workload in sorted(df["workload"].unique()):
            subset = df[df["workload"] == workload]
            mode = subset["mode"].iloc[0]
            if mode == "rate":
                plot_rate_overview(subset, out_dir / f"overview_{workload}.png", f"{title} · {workload}")
            else:
                plot_scaling(subset, out_dir / f"overview_{workload}.png", f"{title} · {workload}")
                plot_frontier(subset, out_dir / f"frontier_{workload}.png", f"{title} · {workload}")

    # Per-axis detail figures, for experiments that varied named knobs.
    for axis in sorted(axis_params):
        rate = _with_baseline(df[df["mode"] == "rate"], axis, by_tag)
        conc = _with_baseline(df[df["mode"] == "concurrent"], axis, by_tag)
        if rate.empty and conc.empty:
            continue
        plot_axis_detail(axis, rate, conc, out_dir / f"axis_{util.sanitize(axis)}.png")

    warn_unusable(df)
    print()
    util.log(f"plots written to {out_dir}")
    util.step(f"two runs side by side, with numbers: ./bench compare <tag> <tag>")
