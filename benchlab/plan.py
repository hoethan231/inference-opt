"""Expand an experiment spec into the ordered list of runs to execute."""

from __future__ import annotations

import dataclasses
import itertools

from . import settings, util, workload_registry
from .spec import Axis, AxisValue, Experiment


@dataclasses.dataclass
class Run:
    """One config to deploy and benchmark."""

    tag: str
    base_tag: str
    suffix: str
    experiment: str
    strategy: str
    model: str
    hf_repo: str
    param: str
    value: str
    label: str
    overrides: dict[str, str]
    extra_args_append: str
    params: dict[str, str]
    extra_args: str
    workloads: list[str]

    def summary(self) -> str:
        return self.label or self.value or self.param


def _compose(
    base_params: dict[str, str],
    base_extra_args: list[str],
    overrides: dict[str, str],
    extra_append: str,
) -> tuple[dict[str, str], str]:
    params = dict(base_params)
    params.update(overrides)
    flags = list(base_extra_args)
    if extra_append:
        flags.append(extra_append)
    extra_args = " ".join(f for f in flags if f).strip()
    params["EXTRA_ARGS"] = extra_args
    return params, extra_args


def _is_baseline_value(value: AxisValue, base_params: dict[str, str]) -> bool:
    """True when this axis value is what the baseline already runs."""
    if value.extra_args:
        return False
    if not value.params:
        return True
    return all(base_params.get(k) == v for k, v in value.params.items())


def _axis_label(axis_key: str | None, param: str, value: AxisValue) -> str:
    return f"{axis_key or param}={value.label}"


def expand(
    experiment: Experiment,
    *,
    suffix: str = "",
    workloads: list[str] | None = None,
    tag_override: str | None = None,
) -> list[Run]:
    base_params, base_extra_args, model = experiment.resolved()
    exp_tag = util.sanitize(experiment.name)
    suffix_tag = util.sanitize(suffix) if suffix else ""
    reserved = len(suffix_tag) + 1 if suffix_tag else 0
    chosen_workloads = workloads or experiment.workloads
    # Fail fast on a bad workload name before anything touches the cluster.
    workload_registry.load_many(chosen_workloads)

    def finish(
        base_tag: str,
        param: str,
        value: str,
        label: str,
        overrides: dict[str, str],
        extra_append: str,
    ) -> Run:
        params, extra_args = _compose(base_params, base_extra_args, overrides, extra_append)
        tag = f"{base_tag}-{suffix_tag}" if suffix_tag else base_tag
        return Run(
            tag=util.fit_name(util.sanitize(tag), settings.MAX_TAG_LEN),
            base_tag=base_tag,
            suffix=suffix,
            experiment=experiment.name,
            strategy=experiment.kind,
            model=model.name,
            hf_repo=model.hf_repo,
            param=param,
            value=value,
            label=label,
            overrides=overrides,
            extra_args_append=extra_append,
            params=params,
            extra_args=extra_args,
            workloads=list(chosen_workloads),
        )

    runs: list[Run] = []

    if experiment.kind == "single":
        base_tag = util.sanitize(tag_override or experiment.name)
        runs.append(finish(base_tag, "single", experiment.name, experiment.name, {}, ""))

    elif experiment.kind == "ofat":
        runs.append(
            finish(f"{exp_tag}-baseline", "baseline", "baseline", "baseline", {}, "")
        )
        for axis in experiment.axes:
            for value in axis.values:
                if _is_baseline_value(value, base_params):
                    continue  # the baseline run already covers this point
                base_tag = _ofat_tag(exp_tag, axis, value.label, reserved)
                runs.append(
                    finish(
                        base_tag,
                        axis.param,
                        value.label,
                        _axis_label(axis.key, axis.param, value),
                        dict(value.params),
                        value.extra_args,
                    )
                )

    else:  # grid
        combos = itertools.product(*[axis.values for axis in experiment.axes])
        for combo in combos:
            overrides: dict[str, str] = {}
            extras: list[str] = []
            label_parts: list[str] = []
            tag_parts: list[str] = []
            for axis, value in zip(experiment.axes, combo):
                overrides.update(value.params)
                if value.extra_args:
                    extras.append(value.extra_args)
                label_parts.append(_axis_label(axis.key, axis.param, value))
                tag_parts.append(f"{axis.tag_abbrev()}-{util.sanitize(value.label)}")
            base_tag = "-".join([exp_tag, *tag_parts])
            runs.append(
                finish(
                    base_tag,
                    "grid",
                    ", ".join(v.label for v in combo),
                    ", ".join(label_parts),
                    overrides,
                    " ".join(extras).strip(),
                )
            )

    _ensure_unique_tags(runs)
    return runs


def _ofat_tag(exp_tag: str, axis: Axis, label: str, reserved: int) -> str:
    """Spell the axis out, but fall back to its abbreviation before truncating.

    A hashed tag is unreadable, and these end up as Kubernetes object names and
    plot labels, so an honest `ab-flashinfer` beats `...backend-9a0b6`. An
    explicit `abbrev:` in the experiment is a deliberate naming choice, so it
    wins outright — that is what keeps a re-run lining up with older tags.
    """
    value = util.sanitize(label)
    if axis.abbrev:
        return f"{exp_tag}-{axis.tag_abbrev()}-{value}"
    spelled = f"{exp_tag}-{util.sanitize(axis.param)}-{value}"
    if len(spelled) <= settings.MAX_TAG_LEN - reserved:
        return spelled
    return f"{exp_tag}-{axis.tag_abbrev()}-{value}"


def _ensure_unique_tags(runs: list[Run]) -> None:
    """Truncation can collide; make tags unique so runs never overwrite results."""
    seen: dict[str, int] = {}
    for run in runs:
        count = seen.get(run.tag, 0)
        seen[run.tag] = count + 1
        if count:
            run.tag = util.fit_name(f"{run.tag}-{count + 1}", settings.MAX_TAG_LEN)


def estimate_seconds(runs: list[Run]) -> int:
    total = 0
    for run in runs:
        total += settings.DEPLOY_OVERHEAD_S
        for name in run.workloads:
            total += workload_registry.load(name).est_seconds
    return total


def describe(runs: list[Run], experiment: Experiment) -> str:
    """The dry-run view: what will run, in what order, and for how long."""
    lines = [
        f"Experiment : {experiment.name}  ({experiment.kind})",
        f"Model      : {experiment.model}",
        f"Workloads  : {', '.join(runs[0].workloads) if runs else '-'}",
    ]
    if experiment.description:
        lines.append(f"Notes      : {experiment.description}")
    if experiment.axes:
        lines.append("Axes       :")
        for axis in experiment.axes:
            values = ", ".join(v.label for v in axis.values)
            lines.append(f"  - {axis.param:<26} {values}")
    lines.append("")
    lines.append(f"Planned runs ({len(runs)}):")
    lines.append(
        util.table(
            ["tag", "param", "value", "overrides"],
            [
                [run.tag, run.param, run.value, _overrides_cell(run)]
                for run in runs
            ],
            indent="  ",
        )
    )
    est = estimate_seconds(runs)
    lines.append("")
    lines.append(
        f"Estimated wall-clock: {util.human_duration(est)} "
        f"({len(runs)} run(s), includes ~{settings.DEPLOY_OVERHEAD_S // 60}m deploy each)"
    )
    return "\n".join(lines)


def _overrides_cell(run: Run) -> str:
    parts = [f"{k}={v}" for k, v in sorted(run.overrides.items())]
    if run.extra_args_append:
        parts.append(f"EXTRA_ARGS += {run.extra_args_append}")
    return ", ".join(parts) or "(base config)"


def filter_runs(runs: list[Run], only: str | None) -> list[Run]:
    """`--only` matches a tag, param name or value, so it's hard to miss with."""
    if not only:
        return runs
    needle = only.lower()
    matched = [
        run
        for run in runs
        if needle in run.tag.lower()
        or needle in run.param.lower()
        or needle in util.sanitize(run.param)
        or needle in run.value.lower()
    ]
    if not matched:
        util.die(f"no planned run matches --only '{only}'")
    return matched


__all__ = ["Run", "describe", "estimate_seconds", "expand", "filter_runs"]
