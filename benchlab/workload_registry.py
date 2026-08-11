"""The workloads/ registry: GuideLLM scenarios plus what shape of load they apply.

The old setup baked one scenario filename into each of two Job templates, so a
third scenario simply couldn't be run. Here a workload is a name, and its shape
(single steady rate vs. a ladder of concurrency levels) is derived from the
scenario JSON, which is also what lets the results loader label rows correctly
without hardcoding "8/16/32/64".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml

from . import settings, util

# guidellm profile kinds that produce one benchmark per run, keyed by the field
# holding their target load.
RATE_KINDS = {"poisson", "constant", "rate", "throughput", "synchronous", "sweep"}
CONCURRENT_KINDS = {"concurrent"}
# The sweep profile generates sweep_size strategies (synchronous, throughput,
# then interpolated rates), so it is treated as rate-oriented rather than a
# single steady rate.
SWEEP_KINDS = {"sweep"}


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    code: str
    file: str
    mode: str  # "rate" | "concurrent" | "sweep"
    targets: list[float]
    est_seconds: int
    note: str = ""

    @property
    def path(self) -> Path:
        return settings.WORKLOADS_DIR / self.file

    @property
    def scenario_name(self) -> str:
        """Key this scenario gets inside the workloads ConfigMap."""
        return self.file


def _profile_shape(scenario: dict[str, Any], path: Path) -> tuple[str, list[float]]:
    profile = ((scenario.get("spec") or {}).get("profile")) or {}
    kind = str(profile.get("kind") or "").lower()
    if kind == "sweep":
        # A sweep runs multiple rate points in one workload execution; model this
        # as N targets so deadline estimation scales with sweep_size.
        try:
            sweep_size = int(profile.get("sweep_size") or 1)
        except (TypeError, ValueError):
            sweep_size = 1
        sweep_size = max(1, sweep_size)
        return "sweep", [float(i) for i in range(1, sweep_size + 1)]
    if kind in CONCURRENT_KINDS:
        streams = profile.get("streams") or profile.get("concurrency") or []
        targets = [float(s) for s in (streams if isinstance(streams, list) else [streams])]
        return "concurrent", targets or [1.0]
    if kind in RATE_KINDS:
        rate = profile.get("rate")
        if rate is None:
            targets = [1.0]
        elif isinstance(rate, list):
            targets = [float(r) for r in rate]
        else:
            targets = [float(rate)]
        return "rate", targets
    if kind in SWEEP_KINDS:
        sweep_size = int(profile.get("sweep_size") or 3)
        return "rate", [float(i) for i in range(sweep_size)]
    util.die(
        f"{path}: unsupported profile kind '{kind}'. Known: "
        f"{', '.join(sorted(RATE_KINDS | CONCURRENT_KINDS | SWEEP_KINDS))}"
    )


def _estimate_seconds(scenario: dict[str, Any], targets: list[float]) -> int:
    spec = scenario.get("spec") or {}
    profile = spec.get("profile") or {}
    per_target = 0.0
    for constraint in spec.get("constraints") or []:
        if constraint.get("kind") == "max_duration":
            per_target += float(constraint.get("seconds") or 0)
    per_target += float(profile.get("warmup") or 0)
    per_target += float(profile.get("rampup_duration") or 0)
    if per_target <= 0:
        per_target = 300.0
    return int(per_target * max(1, len(targets)))


def _load_index() -> dict[str, dict[str, Any]]:
    if not settings.WORKLOADS_INDEX.exists():
        util.die(f"missing workload registry: {settings.WORKLOADS_INDEX}")
    raw = yaml.safe_load(settings.WORKLOADS_INDEX.read_text()) or {}
    entries = raw.get("workloads") or {}
    if not entries:
        util.die(f"{settings.WORKLOADS_INDEX}: no workloads defined")
    return entries


def available() -> list[str]:
    return sorted(_load_index())


def load(name: str) -> Workload:
    entries = _load_index()
    entry = entries.get(name)
    if entry is None:
        util.die(f"unknown workload '{name}'. Available: {', '.join(sorted(entries))}")
    filename = entry.get("file")
    if not filename:
        util.die(f"{settings.WORKLOADS_INDEX}: workload '{name}' has no 'file'")
    path = settings.WORKLOADS_DIR / filename
    if not path.exists():
        util.die(f"workload '{name}' points at a missing scenario file: {path}")
    try:
        scenario = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        util.die(f"{path}: invalid JSON ({exc})")
    mode, targets = _profile_shape(scenario, path)
    return Workload(
        name=name,
        code=util.sanitize(entry.get("code") or name)[:6],
        file=filename,
        mode=mode,
        targets=targets,
        est_seconds=_estimate_seconds(scenario, targets),
        note=str(entry.get("note") or "").strip(),
    )


def load_many(names: list[str]) -> list[Workload]:
    return [load(n) for n in names]


def all_scenario_files() -> list[Path]:
    """Every scenario referenced by the registry, for the ConfigMap."""
    return [load(name).path for name in available()]
