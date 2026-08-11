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

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: everything is written to PNG

import matplotlib.pyplot as plt
import pandas as pd

from . import ingest, settings, util
from .store import RunRecord

CMAP = plt.get_cmap("tab10")
BASELINE_COLOR = "#d62728"
ANNOT_BBOX = dict(boxstyle="round,pad=0.22", fc="white", ec="#333333", alpha=0.92, lw=0.8)

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
    ("tps_p50", "Total throughput p50, tokens/s"),
    ("tps_p99", "Total throughput p99, tokens/s"),
    ("output_tps_p50", "Output throughput p50, tokens/s"),
    ("output_tps_p99", "Output throughput p99, tokens/s"),
    ("rps_mean", "Request completion rate avg, req/s"),
    ("error_rate_pct", "Error rate (%)"),
]

DETAIL_P50 = [
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("itl_ms_p50", "ITL p50 (ms)"),
    ("tps_p50", "Total throughput p50, tokens/s"),
    ("output_tps_p50", "Output throughput p50, tokens/s"),
    ("rps_mean", "Request completion rate avg, req/s"),
    ("error_rate_pct", "Error rate (%)"),
]
DETAIL_P99 = [
    ("ttft_ms_p99", "TTFT p99 (ms)"),
    ("itl_ms_p99", "ITL p99 (ms)"),
    ("tps_p99", "Total throughput p99, tokens/s"),
    ("output_tps_p99", "Output throughput p99, tokens/s"),
    ("rps_mean", "Request completion rate avg, req/s"),
    ("error_rate_pct", "Error rate (%)"),
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


def _with_report_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error_rate_pct"] = pd.to_numeric(out["error_rate"], errors="coerce") * 100.0
    return out


def _fmt_metric_value(value: object) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(num):
        return ""
    if abs(num) >= 1000:
        return f"{num:,.0f}"
    if abs(num) >= 100:
        return f"{num:.1f}"
    if abs(num) >= 10:
        return f"{num:.2f}"
    return f"{num:.3f}"


def plot_rate_overview(df: pd.DataFrame, path: Path, title: str) -> None:
    """Every run compared head-to-head at a fixed offered load."""
    df = _with_report_metrics(df)
    ordered = df.sort_values("ttft_ms_p50")
    labels = list(ordered["series"])
    baselines = set(df[df["param"] == "baseline"]["series"])
    height = max(6.0, 0.42 * len(labels) + 4)
    cols = 2
    rows = int(math.ceil(len(OVERVIEW_METRICS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, max(height, 3.3 * rows)))
    flat_axes = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, (column, label) in zip(flat_axes, OVERVIEW_METRICS):
        bars = ax.barh(labels, ordered[column], color=_colors(labels, baselines))
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)
        for bar, value in zip(bars, ordered[column].tolist()):
            text = _fmt_metric_value(value)
            if text:
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2.0,
                    f" {text}",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    fontweight="bold",
                    bbox=ANNOT_BBOX,
                )
    for ax in list(flat_axes)[len(OVERVIEW_METRICS) :]:
        ax.axis("off")
    fig.suptitle(f"{title} — steady offered load", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)


def plot_scaling(
    df: pd.DataFrame, path: Path, title: str, *, target_label: str = "concurrent users"
) -> None:
    """How each run's latency and throughput move with target load."""
    df = _with_report_metrics(df)
    order = sorted(df["series"].unique(), key=_sort_key(df))
    baselines = set(df[df["param"] == "baseline"]["series"])
    cols = 2
    rows = int(math.ceil(len(OVERVIEW_METRICS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.4 * rows))
    flat_axes = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, (column, label) in zip(flat_axes, OVERVIEW_METRICS):
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
        ax.set_xlabel(target_label)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    for ax in list(flat_axes)[len(OVERVIEW_METRICS) :]:
        ax.axis("off")
    handles, labels = list(flat_axes)[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, max(1, len(order))),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(f"{title} — scaling with {target_label}", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    _save(fig, path)


def plot_frontier(
    df: pd.DataFrame, path: Path, title: str, *, point_label: str = "target"
) -> None:
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
    fig.suptitle(
        f"{title} — latency/throughput frontier (points labelled with {point_label})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def plot_sweep_rps_detail(df: pd.DataFrame, path: Path, title: str) -> None:
    """Explicit per-RPS p50/p99 charts for sweep workloads."""
    df = _with_report_metrics(df)
    order = sorted(df["series"].unique(), key=_sort_key(df))
    baselines = set(df[df["param"] == "baseline"]["series"])
    fig, axes = plt.subplots(5, 2, figsize=(16, 19))
    panels = [
        ("ttft_ms_p50", "TTFT p50 (ms)"),
        ("ttft_ms_p99", "TTFT p99 (ms)"),
        ("itl_ms_p50", "ITL p50 (ms)"),
        ("itl_ms_p99", "ITL p99 (ms)"),
        ("tps_p50", "Total throughput p50, tokens/s"),
        ("tps_p99", "Total throughput p99, tokens/s"),
        ("output_tps_p50", "Generation throughput p50, tokens/s"),
        ("output_tps_p99", "Generation throughput p99, tokens/s"),
        ("rps_mean", "Request completion rate avg, req/s"),
    ]
    for ax, (column, label) in zip(axes.flat[:9], panels):
        for i, series in enumerate(order):
            subset = df[df["series"] == series].sort_values("target")
            is_base = series in baselines
            ax.plot(
                subset["target"],
                subset[column],
                marker="o",
                label=series,
                color=BASELINE_COLOR if is_base else CMAP(i % 10),
                linewidth=2.5 if is_base else 1.8,
            )
            for _, row in subset.iterrows():
                value = row[column]
                if pd.notna(value):
                    ax.annotate(
                        f"{float(value):.1f}",
                        (row["target"], value),
                        fontsize=8.5,
                        fontweight="bold",
                        textcoords="offset points",
                        xytext=(8, 7),
                        bbox=ANNOT_BBOX,
                    )
        ax.set_xlabel("offered load (req/s)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, max(1, len(order))),
            fontsize=7,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle(f"{title} — per-RPS sweep metrics (values labelled)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, path)


def write_sweep_rps_table(df: pd.DataFrame, path: Path) -> None:
    """CSV with one row per sweep point keyed by offered load (req/s)."""
    columns = [
        "series",
        "target",
        "ttft_ms_p50",
        "ttft_ms_p99",
        "itl_ms_p50",
        "itl_ms_p99",
        "tps_p50",
        "tps_p99",
        "output_tps_p50",
        "output_tps_p99",
        "rps_mean",
        "requests_successful",
        "error_rate",
    ]
    table = df[columns].copy().sort_values(["series", "target"])
    table = table.rename(columns={"target": "offered_rps"})
    table["error_rate_pct"] = table["error_rate"] * 100.0
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    try:
        shown = path.relative_to(settings.REPO)
    except ValueError:
        shown = path
    util.step(f"wrote {shown}")


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
    if (
        "derived_axis" in rows.columns
        and rows["derived_axis"].any()
        and "experiment" in rows.columns
        and rows["experiment"].eq("external").all()
    ):
        rows = _aggregate_external_axis(rows)
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
        # Relabelled from the same rules as every other series, so a baseline on a
        # different model or GPU topology says so instead of blending in.
        prefixes = ingest.label_prefixes(df)
        baseline["series"] = baseline.apply(
            lambda r: ingest.series_label(r, prefixes, r["value_display"]), axis=1
        )

    combined = pd.concat([baseline, rows], ignore_index=True)
    return combined.drop_duplicates(subset=["series", "workload", "target"])


def _aggregate_external_axis(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse external grid points into one sweep value per axis value."""
    grouped = rows.copy()
    grouped["series"] = grouped["value_display"]
    keys = ["series", "workload", "mode", "target", "param", "value", "value_display"]
    numeric = grouped.select_dtypes(include="number").columns.tolist()
    aggregated = grouped.groupby(keys, as_index=False)[numeric].mean()
    aggregated["tag"] = "external-aggregate"
    aggregated["suffix"] = ""
    aggregated["derived_axis"] = True
    return aggregated


def plot_axis_detail(
    param: str, rate: pd.DataFrame, conc: pd.DataFrame, path: Path
) -> pd.DataFrame | None:
    """Bar charts at steady load over scaling lines, p50 above p99."""
    if rate.empty and conc.empty:
        return None
    rate = _with_report_metrics(rate) if not rate.empty else rate
    conc = _with_report_metrics(conc) if not conc.empty else conc
    reference = rate if not rate.empty else conc
    order = sorted(reference["series"].unique(), key=_sort_key(reference))
    baselines = (
        set(reference[reference["is_baseline"]]["series"])
        if "is_baseline" in reference
        else set()
    )

    cols = max(len(DETAIL_P50), len(DETAIL_P99))
    rows = 4 if not conc.empty else 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.7 * rows))
    axes = axes.reshape(rows, cols)

    def bars(axes_row, metrics):
        indexed = rate.drop_duplicates("series").set_index("series").reindex(order)
        for ax, (column, label) in zip(axes_row, metrics):
            bars = ax.bar(order, indexed[column], color=_colors(order, baselines))
            ax.set_title(label)
            ax.set_xlabel(param)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.grid(axis="y", alpha=0.3)
            for bar, value in zip(bars, indexed[column].tolist()):
                text = _fmt_metric_value(value)
                if text:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        text,
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        fontweight="bold",
                        bbox=ANNOT_BBOX,
                    )
        for ax in axes_row[len(metrics) :]:
            ax.axis("off")

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
        for ax in axes_row[len(metrics) :]:
            ax.axis("off")
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
    overview_df = df[~df["derived_axis"]] if "derived_axis" in df.columns else df
    out_dir.mkdir(parents=True, exist_ok=True)

    # "unknown" is what an import gets when no axis was ever recorded for it: a
    # bag of unrelated one-offs, which is not an axis and can't be plotted as one.
    axis_source = df
    axis_params = [
        p
        for p in axis_source["param"].unique()
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
        for workload in sorted(overview_df["workload"].unique()):
            subset = overview_df[overview_df["workload"] == workload]
            mode = subset["mode"].iloc[0]
            if mode == "rate":
                if subset["target"].nunique() > 1:
                    plot_scaling(
                        subset,
                        out_dir / f"overview_{workload}.png",
                        f"{title} · {workload}",
                        target_label="offered load (req/s)",
                    )
                    plot_frontier(
                        subset,
                        out_dir / f"frontier_{workload}.png",
                        f"{title} · {workload}",
                        point_label="offered req/s",
                    )
                    plot_sweep_rps_detail(
                        subset,
                        out_dir / f"sweep_rps_{workload}.png",
                        f"{title} · {workload}",
                    )
                    write_sweep_rps_table(subset, out_dir / f"sweep_rps_{workload}.csv")
                else:
                    plot_rate_overview(
                        subset, out_dir / f"overview_{workload}.png", f"{title} · {workload}"
                    )
            elif mode == "concurrent":
                plot_scaling(subset, out_dir / f"overview_{workload}.png", f"{title} · {workload}")
                plot_frontier(subset, out_dir / f"frontier_{workload}.png", f"{title} · {workload}")
            elif mode == "sweep":
                plot_scaling(
                    subset,
                    out_dir / f"overview_{workload}.png",
                    f"{title} · {workload}",
                    target_label="offered load (req/s)",
                )
                plot_frontier(
                    subset,
                    out_dir / f"frontier_{workload}.png",
                    f"{title} · {workload}",
                    point_label="offered req/s",
                )
                plot_sweep_rps_detail(
                    subset,
                    out_dir / f"sweep_rps_{workload}.png",
                    f"{title} · {workload}",
                )
                write_sweep_rps_table(subset, out_dir / f"sweep_rps_{workload}.csv")
            else:
                util.warn(
                    f"unknown workload mode '{mode}' for '{workload}'; using rate overview"
                )
                plot_rate_overview(subset, out_dir / f"overview_{workload}.png", f"{title} · {workload}")

    # Per-axis detail figures, for experiments that varied named knobs.
    for axis in sorted(axis_params):
        axis_rows = axis_source[axis_source["param"] == axis]
        contexts = (
            sorted(
                {
                    str(v)
                    for v in axis_rows.get("axis_context", pd.Series(dtype=str)).unique()
                    if str(v).strip()
                }
            )
            if "axis_context" in axis_rows.columns
            else []
        )
        if not contexts:
            contexts = [""]

        for context in contexts:
            scoped = axis_source
            if context:
                scoped = axis_source[axis_source.get("axis_context", "") == context]
            rate = _with_baseline(scoped[scoped["mode"] == "rate"], axis, by_tag)
            conc = _with_baseline(scoped[scoped["mode"] == "concurrent"], axis, by_tag)
            if rate.empty and conc.empty:
                continue
            # For grid slices (for example "axis @ other_axis=value"), the title
            # already carries the conditioning context. Keep series labels as the
            # swept value only to avoid noisy tag suffixes on the x-axis.
            if context:
                if not rate.empty:
                    rate = rate.copy()
                    rate["series"] = rate["value_display"].astype(str)
                if not conc.empty:
                    conc = conc.copy()
                    conc["series"] = conc["value_display"].astype(str)
            stem = f"axis_{util.sanitize(axis)}"
            axis_title = axis
            if context:
                stem += f"__at_{util.sanitize(context)}"
                axis_title = f"{axis} @ {context}"
            plot_axis_detail(axis_title, rate, conc, out_dir / f"{stem}.png")

    warn_unusable(df)
    print()
    util.log(f"plots written to {out_dir}")
    util.step(f"two runs side by side, with numbers: ./bench compare <tag> <tag>")
