"""Head-to-head comparison of two runs.

`bench report` draws every run at once, which is the wrong tool when the
question is just "is B better than A" — a candidate against the baseline, or two
finalists against each other. This prints the deltas and draws only those two.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ingest, report, store, util
from .store import RunRecord

# (column, label, digits, lower_is_better)
DELTA_METRICS = [
    ("ttft_ms_p50", "TTFT p50 (ms)", 1, True),
    ("ttft_ms_p99", "TTFT p99 (ms)", 1, True),
    ("itl_ms_p50", "ITL p50 (ms)", 2, True),
    ("itl_ms_p99", "ITL p99 (ms)", 2, True),
    ("latency_s_p50", "E2E latency p50 (s)", 2, True),
    ("latency_s_p99", "E2E latency p99 (s)", 2, True),
    ("tps_mean", "Throughput (tok/s)", 1, False),
    ("error_rate", "Error rate", 4, True),
]


def _delta_rows(
    left: pd.Series, right: pd.Series
) -> list[list[str]]:
    rows = []
    for column, label, digits, lower_better in DELTA_METRICS:
        a, b = left.get(column), right.get(column)
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            rows.append([label, util.fmt(a, digits), util.fmt(b, digits), "-", "-"])
            continue
        if a == 0:
            # A percentage change from zero is meaningless; show the absolute move.
            change_text = f"{b - a:+,.{digits}f} abs"
            direction = b - a
        else:
            percent = (b - a) / a * 100
            change_text = f"{percent:+.1f}%"
            direction = percent
        if abs(direction) < 1e-12:
            verdict = "same"
        else:
            improved = direction < 0 if lower_better else direction > 0
            verdict = "better" if improved else "worse"
        rows.append([label, util.fmt(a, digits), util.fmt(b, digits), change_text, verdict])
    return rows


def _plot_bars(
    df: pd.DataFrame, labels: tuple[str, str], path: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    metrics = [m for m in DELTA_METRICS]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (column, label, _, _) in zip(axes.flat, metrics):
        values = [
            df[df["series_label"] == labels[0]][column].mean(),
            df[df["series_label"] == labels[1]][column].mean(),
        ]
        ax.bar(list(labels), values, color=["#1f77b4", "#ff7f0e"])
        ax.set_title(label)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    report._save(fig, path)


def _plot_lines(
    df: pd.DataFrame, labels: tuple[str, str], path: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (column, label, _, _) in zip(axes.flat, DELTA_METRICS):
        for name, color in zip(labels, ("#1f77b4", "#ff7f0e")):
            subset = df[df["series_label"] == name].sort_values("target")
            ax.plot(subset["target"], subset[column], marker="o", label=name, color=color)
        ax.set_xlabel("concurrent users")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    report._save(fig, path)


def run(
    tag_a: str,
    tag_b: str,
    out_dir: Path,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
) -> None:
    record_a = store.resolve_tag(tag_a)
    record_b = store.resolve_tag(tag_b)
    if record_a.tag == record_b.tag:
        util.die("both arguments resolved to the same run")

    labels = (
        label_a or _default_label(record_a),
        label_b or _default_label(record_b),
    )
    df = ingest.frame([record_a, record_b])
    df["series_label"] = df["tag"].map({record_a.tag: labels[0], record_b.tag: labels[1]})

    for workload in sorted(df["workload"].unique()):
        subset = df[df["workload"] == workload]
        present = set(subset["series_label"])
        if len(present) < 2:
            util.warn(f"workload '{workload}' only has results for {', '.join(present)}; skipping")
            continue
        mode = subset["mode"].iloc[0]
        title = f"{labels[0]} vs {labels[1]} · {workload}"
        stem = f"{util.sanitize(record_a.tag)}_vs_{util.sanitize(record_b.tag)}_{workload}"
        if mode == "rate":
            _plot_bars(subset, labels, out_dir / f"{stem}.png", title)
        else:
            _plot_lines(subset, labels, out_dir / f"{stem}.png", title)
            report.plot_frontier(
                subset.assign(series=subset["series_label"], param=""),
                out_dir / f"{stem}_frontier.png",
                title,
            )

        left = subset[subset["series_label"] == labels[0]].mean(numeric_only=True)
        right = subset[subset["series_label"] == labels[1]].mean(numeric_only=True)
        scope = "" if mode == "rate" else " (mean over concurrency levels)"
        print()
        util.rule(f"{workload}: {labels[1]} vs {labels[0]}{scope}")
        print(
            util.table(
                ["metric", labels[0], labels[1], "change", "verdict"],
                _delta_rows(left, right),
            )
        )

    _print_config_diff(record_a, record_b, labels)
    util.log(f"plots written to {out_dir}")


def _default_label(record: RunRecord) -> str:
    if record.param in ("baseline", "single"):
        return record.tag
    return f"{record.tag}"


def _print_config_diff(a: RunRecord, b: RunRecord, labels: tuple[str, str]) -> None:
    def settings_of(record: RunRecord) -> dict[str, str]:
        merged = dict(record.params)
        for key, value in (
            ("EXTRA_ARGS", record.extra_args),
            ("EXTRA_ARGS_APPEND", record.extra_args_append),
        ):
            if value:
                merged[key] = value
        return merged

    left, right = settings_of(a), settings_of(b)
    print()
    if not left or not right:
        missing = [label for label, cfg in zip(labels, (left, right)) if not cfg]
        util.warn(
            f"no config recorded for {' and '.join(missing)}, so the difference "
            f"cannot be shown — check the run's log if you need it"
        )
        return

    rows = []
    for key in sorted(set(left) | set(right)):
        before, after = left.get(key, ""), right.get(key, "")
        if before == after:
            continue
        if key.startswith("EXTRA_ARGS"):
            # These are long flag strings; the interesting part is which flags
            # came and went, not the two paragraphs they sit in.
            rows.extend(_flag_rows(key, before, after))
        else:
            rows.append([key, before or "-", after or "-"])

    # An imported run has no config of its own on record: what we hold is today's
    # params file, so an empty diff there means "unknown", not "the same".
    guessed = [label for label, rec in zip(labels, (a, b)) if _config_is_guessed(rec)]
    if rows:
        util.rule("config difference")
        print(util.table(["key", labels[0], labels[1]], rows))
    elif guessed:
        util.warn(
            f"the config shown for {' and '.join(guessed)} was reconstructed on import "
            f"and is not necessarily what ran, so no difference can be trusted here"
        )
        return
    else:
        util.step("the two runs used an identical config (so this is run-to-run variance)")
        return
    if guessed:
        util.step(f"config for {' and '.join(guessed)} was reconstructed on import")


def _flag_rows(key: str, before: str, after: str) -> list[list[str]]:
    """One row per flag that differs, so a long EXTRA_ARGS stays readable."""
    left = {flag.split("=", 1)[0]: flag for flag in before.split()}
    right = {flag.split("=", 1)[0]: flag for flag in after.split()}
    rows = []
    for name in sorted(set(left) | set(right)):
        if left.get(name) == right.get(name):
            continue
        rows.append(
            [
                f"{key} {name}",
                util.elide(left.get(name, "(absent)"), 44),
                util.elide(right.get(name, "(absent)"), 44),
            ]
        )
    return rows


def _config_is_guessed(record: RunRecord) -> bool:
    """True for imported runs whose real config was never recorded anywhere."""
    return not record.overrides and "reconstructed" in (record.note or "")
