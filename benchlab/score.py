"""Ranking: a composite score with a hard reliability gate.

The gate matters more than the score, in two ways.

A config that errors out most of its requests looks *fast*, because only the
lucky survivors get measured — so error rate is not blended into the score as one
term among several. It disqualifies.

A config that completed *no* requests looks perfect: zero latency, zero errors.
Several such runs exist in the imported history (a context window too small for
the workload's prefixes produces an empty report) and they used to rank first.
A run has to have measured something to be ranked at all.
"""

from __future__ import annotations

import pandas as pd

# (column, direction of "better")
SCORING_METRICS = [
    ("ttft_ms_p90", "min"),
    ("latency_s_p90", "min"),
    ("tps_mean", "max"),
]

ERROR_RATE_THRESHOLD = 0.05

RANK_COLUMNS = [
    "ttft_ms_p90",
    "latency_s_p90",
    "tps_mean",
    "error_rate",
    "requests_total",
    "requests_successful",
]


def score(rows: pd.DataFrame, *, error_threshold: float = ERROR_RATE_THRESHOLD) -> pd.DataFrame:
    """Add `measured`, `reliable` and `composite_score`, sorted best first."""
    scored = rows.copy()
    error_rate = pd.to_numeric(scored.get("error_rate"), errors="coerce")
    successful = pd.to_numeric(scored.get("requests_successful"), errors="coerce").fillna(0)
    total = pd.to_numeric(scored.get("requests_total"), errors="coerce").fillna(0)

    scored["measured"] = (total > 0) & (successful > 0)
    # An unknown error rate counts against a run: we can't verify it.
    scored["reliable"] = scored["measured"] & (error_rate.fillna(1.0) <= error_threshold)

    parts = []
    for column, direction in SCORING_METRICS:
        if column not in scored:
            continue
        values = pd.to_numeric(scored[column], errors="coerce")
        low, high = values.min(), values.max()
        if pd.isna(low) or pd.isna(high) or high == low:
            normalised = pd.Series(1.0, index=values.index)
        else:
            normalised = (values - low) / (high - low)
            if direction == "min":
                normalised = 1 - normalised
        parts.append(normalised.fillna(0.0))

    scored["composite_score"] = (
        pd.concat(parts, axis=1).mean(axis=1) if parts else 0.0
    )
    # Unreliable rows sort last however good their surviving requests looked;
    # unmeasured rows sort below those, since they say nothing at all.
    scored.loc[~scored["reliable"], "composite_score"] = -1.0
    scored.loc[~scored["measured"], "composite_score"] = -2.0
    return scored.sort_values("composite_score", ascending=False)


def ranking_basis(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """One row per run to rank, preferring a steady-rate workload.

    Concurrency ladders give several rows per run, so they are averaged when
    that's all we have.
    """
    rate = df[df["mode"] == "rate"]
    if not rate.empty:
        workload = sorted(rate["workload"].unique())[0]
        single = rate[rate["workload"] == workload]
        return single.drop_duplicates(subset=["tag"]), f"workload '{workload}'"

    columns = [c for c in RANK_COLUMNS if c in df.columns]
    keys = ["tag", "series", "value_display", "model", "param", "value", "label"]
    keys = [k for k in keys if k in df.columns]
    grouped = df.groupby(keys, as_index=False)[columns].mean()
    workload = sorted(df["workload"].unique())[0]
    return grouped, f"workload '{workload}' (mean over concurrency levels)"
