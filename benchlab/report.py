"""Plots and leaderboards for whatever has run.

Replaces plot_results.py (OFAT), plot_grid.py (grid, which imported half of
plot_results) and their two ranking CSVs. There is one code path: the shape of
the data decides what gets drawn. A steady-rate workload becomes bar charts, a
concurrency ladder becomes scaling lines plus a latency/throughput frontier, and
an experiment with named axes additionally gets a per-axis detail figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: everything is written to PNG

import matplotlib.pyplot as plt
import pandas as pd

import yaml

from . import ingest, score, settings, spec, store, util
from .store import RunRecord

CMAP = plt.get_cmap("tab10")
BASELINE_COLOR = "#d62728"

OVERVIEW_METRICS = [
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("ttft_ms_p99", "TTFT p99 (ms)"),
    ("itl_ms_p50", "ITL p50 (ms)"),
    ("itl_ms_p99", "ITL p99 (ms)"),
    ("tps_mean", "Throughput, tokens/s (mean)"),
    ("error_rate", "Error rate"),
]

DETAIL_P50 = [
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("itl_ms_p50", "ITL p50 (ms)"),
    ("tps_p50", "Throughput p50, tokens/s"),
]
DETAIL_P99 = [
    ("ttft_ms_p99", "TTFT p99 (ms)"),
    ("itl_ms_p99", "ITL p99 (ms)"),
    ("tps_p99", "Throughput p99, tokens/s"),
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
    for ax, column, label in (
        (axes[0], "ttft_ms_p50", "TTFT p50 (ms)"),
        (axes[1], "ttft_ms_p99", "TTFT p99 (ms)"),
    ):
        for i, series in enumerate(order):
            subset = df[df["series"] == series].sort_values("target")
            is_base = series in baselines
            ax.plot(
                subset["tps_mean"],
                subset[column],
                marker="o",
                label=series,
                color=BASELINE_COLOR if is_base else CMAP(i % 10),
                linewidth=2.5 if is_base else 1.5,
            )
            for _, row in subset.iterrows():
                ax.annotate(
                    str(int(row["target"])),
                    (row["tps_mean"], row[column]),
                    fontsize=7,
                    textcoords="offset points",
                    xytext=(4, 4),
                )
        ax.set_xlabel("Throughput, tokens/s (mean)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title} — latency/throughput frontier (points labelled with concurrency)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def plot_ranking(board: pd.DataFrame, path: Path, title: str, top: int | None) -> None:
    shown = board.head(top) if top else board
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.45 * len(shown) + 2)))
    colors = ["#2ca02c" if ok else BASELINE_COLOR for ok in shown["reliable"]]
    ax.barh(shown["series"], shown["composite_score"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("composite score (higher is better; red failed the error-rate gate)")
    ax.set_title(f"{title} — ranking")
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)
    for i, (_, row) in enumerate(shown.iterrows()):
        ax.annotate(
            f"{row['composite_score']:.3f}",
            (row["composite_score"], i),
            va="center",
            fontsize=8,
            xytext=(3, 0),
            textcoords="offset points",
        )
    fig.tight_layout()
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


def leaderboard(rows: pd.DataFrame, *, top: int | None = None) -> pd.DataFrame:
    board = score.score(rows).reset_index(drop=True)
    board.insert(0, "rank", board.index + 1)
    return board.head(top) if top else board


def print_leaderboard(board: pd.DataFrame, title: str, basis: str) -> None:
    util.rule(f"{title}  ({basis}; error rate > {score.ERROR_RATE_THRESHOLD:.0%} disqualified)")
    columns = [
        ("rank", "rank", 0),
        ("series", "config", None),
        ("verdict", "ok", None),
        ("requests_successful", "reqs ok", 0),
        ("ttft_ms_p90", "ttft p90 ms", 0),
        ("latency_s_p90", "e2e p90 s", 1),
        ("tps_mean", "tok/s", 0),
        ("error_rate", "err", 3),
        ("composite_score", "score", 3),
    ]
    board = board.copy()
    board["verdict"] = [
        "yes" if reliable else ("no data" if not measured else "HIGH ERR")
        for reliable, measured in zip(board["reliable"], board["measured"])
    ]
    headers = [header for column, header, _ in columns if column in board]
    rows = []
    for _, row in board.iterrows():
        cells = []
        for column, _, digits in columns:
            if column not in board:
                continue
            value = row[column]
            cells.append(str(value) if digits is None else util.fmt(value, digits))
        rows.append(cells)
    print(util.table(headers, rows))

    unmeasured = board[~board["measured"]]
    if not unmeasured.empty:
        util.warn(
            f"{len(unmeasured)} config(s) completed no requests at all and cannot be "
            f"ranked: {', '.join(str(s) for s in unmeasured['series'].head(6))}"
        )


def _overrides_summary(overrides: dict[str, str], *, said: str, limit: int = 4) -> str:
    """What the winner set beyond what its own label already says.

    Imported grid runs recorded their entire base config here, so this would
    otherwise print a paragraph of keys the label already covers.
    """
    items = [
        f"{key}={value}"
        for key, value in sorted((overrides or {}).items())
        if key not in said and len(value) < 60
    ]
    if not items:
        return ""
    hidden = len(overrides) - len(items[:limit])
    return ", ".join(items[:limit]) + (f" (+{hidden} more)" if hidden > 0 else "")


def write_recommended(
    winners: list[tuple[str, RunRecord]],
    path: Path,
    source: str,
) -> Path | None:
    """Write the winning configuration as a runnable single-config experiment.

    The old plot_results.py wrote this as a ready-to-apply ConfigMap, which meant
    the winning combination was never itself measured. Here it comes out as an
    experiment so the obvious next step — actually running it — is one command,
    and its result lands in the store like any other.

    With one winner (a grid, where the combination really was measured) that is
    simply a copy. With one winner per axis (OFAT) it is a hypothesis, since each
    knob was chosen in isolation.
    """
    models = {r.model for _, r in winners}
    if len(models) > 1:
        util.step("skipping the recommended config: these winners span several models")
        return None

    params: dict[str, str] = {}
    flags: list[str] = []
    contested: dict[str, list[str]] = {}
    for axis, record in winners:
        for key, value in (record.overrides or {}).items():
            # Identity comes from the model registry and EXTRA_ARGS is composed,
            # so neither may be written back as a param. Imported runs recorded
            # both, which would make this file fail to load.
            if key in spec.RESERVED_PARAM_KEYS:
                continue
            if key in params and params[key] != value:
                contested.setdefault(key, [f"{params[key]}"]).append(f"{value} ({axis})")
            params[key] = value
        if record.extra_args_append and record.extra_args_append not in flags:
            flags.append(record.extra_args_append)

    if not params and not flags:
        util.step("the baseline won, so there is no new config to recommend")
        return None

    for key, values in contested.items():
        util.warn(
            f"axes disagree on {key} ({' vs '.join(values)}); kept the last. "
            f"OFAT cannot see interactions — confirm with a small grid"
        )

    combined = len(winners) > 1
    provenance = (
        f"Per-axis winners from '{source}', combined. Untested as a combination: "
        f"OFAT picks each knob in isolation, so run it and compare before "
        f"trusting it."
        if combined
        else f"The winning config of '{source}' ({winners[0][1].tag}), as measured."
    )
    model = next(iter(models))
    document = {
        "name": path.stem,
        "description": (
            f"{provenance} Written by ./bench report on {store.utc_now()}."
        ),
        "model": model,
        "workloads": sorted({w for _, r in winners for w in (r.workloads or [])})
        or ["rate", "concurrent"],
        "params": params,
        "strategy": {"kind": "single"},
    }
    if flags:
        document["extra_args"] = flags

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        f"# {axis}: {record.tag}" for axis, record in sorted(winners, key=lambda w: w[0])
    )
    path.write_text(
        f"# Auto-written; edit freely or delete. Winning run per "
        f"{'axis' if combined else 'experiment'}:\n"
        f"{header}\n"
        + yaml.safe_dump(document, sort_keys=False, width=4096)
    )
    try:
        shown = path.relative_to(settings.REPO)
    except ValueError:
        shown = path
    util.ok(
        f"recommended config "
        f"{'from the per-axis winners' if combined else 'from the winning run'} "
        f"-> {shown}"
    )
    util.step(f"{'measure it' if combined else 'reuse it'}: ./bench run {shown}")
    return path


def build(
    records: list[RunRecord],
    out_dir: Path,
    title: str,
    *,
    param: str | None = None,
    top: int | None = None,
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
    winners: list[tuple[str, RunRecord]] = []
    for axis in sorted(axis_params):
        rate = _with_baseline(df[df["mode"] == "rate"], axis, by_tag)
        conc = _with_baseline(df[df["mode"] == "concurrent"], axis, by_tag)
        if rate.empty and conc.empty:
            continue
        plot_axis_detail(axis, rate, conc, out_dir / f"axis_{util.sanitize(axis)}.png")
        combined = pd.concat([f for f in (rate, conc) if not f.empty], ignore_index=True)
        basis_rows, basis = score.ranking_basis(combined)
        board = leaderboard(basis_rows)
        print()
        print_leaderboard(board, f"AXIS: {axis}", basis)
        won = board[board["reliable"]]
        if not won.empty:
            record = by_tag.get(won.iloc[0]["tag"])
            if record:
                winners.append((axis, record))

    if param:
        util.log(f"plots written to {out_dir}")
        return

    # Overall ranking across every run in view.
    basis_rows, basis = score.ranking_basis(df)
    board = leaderboard(basis_rows)
    print()
    print_leaderboard(board, title.upper(), basis)
    plot_ranking(board, out_dir / "ranking.png", title, top)

    reliable = board[board["reliable"]]
    print()
    if reliable.empty:
        util.warn(
            f"every run exceeded the {score.ERROR_RATE_THRESHOLD:.0%} error-rate gate — "
            f"treat these latency numbers as survivor-biased and investigate first"
        )
    else:
        best = reliable.iloc[0]
        util.ok(f"best: {best['series']}  (composite {best['composite_score']:.3f})")
        record = by_tag.get(best["tag"]) if "tag" in best else None
        if record:
            extra = _overrides_summary(record.overrides, said=str(best["series"]))
            if extra:
                print(f"    also set: {extra}")
            print(f"    confirm it head-to-head: ./bench compare <baseline-tag> {record.tag}")
            # A grid winner is a whole configuration that really ran, so it can be
            # recommended on its own; an OFAT sweep needs its axes combined first.
            if not winners and record.overrides:
                winners = [(record.param, record)]

    # A recommendation only means something when one experiment's configs were
    # compared against each other: a mixed view is for comparing, not choosing,
    # and a view of one run has nothing to choose between.
    single_experiment = len({r.experiment for r in records}) == 1
    if winners and single_experiment and len(board) > 1:
        print()
        write_recommended(
            winners, out_dir / f"recommended-{util.sanitize(title)}.yaml", title
        )
    util.log(f"plots written to {out_dir}")
