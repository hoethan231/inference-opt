"""Experiment specs: one YAML format for single runs, OFAT sweeps and grids.

The three drivers this replaces each had their own notion of a plan — a hand
edited ConfigMap, a Python `SWEEP` list, and a grid JSON. They are the same
thing at different arities, so they share one schema here:

    kind: single   ->  exactly one config
    kind: ofat     ->  baseline + one axis value changed at a time (linear cost)
    kind: grid     ->  the full Cartesian product of the axes (multiplicative)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from . import model_registry, settings, util

KINDS = ("single", "ofat", "grid")

# A knob set to the empty string means "let vLLM decide" (the attention backend
# is the usual case). Empty strings make terrible tags and plot labels, so they
# are displayed as this instead.
AUTO_LABEL = "auto"

# Composed from `extra_args:` lists; setting it directly would silently discard
# the model's own flags, which is how configs used to drift apart.
RESERVED_PARAM_KEYS = {"EXTRA_ARGS", "MODEL_NAME", "SERVED_MODEL_NAME"}


@dataclasses.dataclass(frozen=True)
class AxisValue:
    """One point on an axis: a display label plus what it changes."""

    label: str
    params: dict[str, str] = dataclasses.field(default_factory=dict)
    extra_args: str = ""


@dataclasses.dataclass(frozen=True)
class Axis:
    param: str
    values: list[AxisValue]
    key: str | None = None
    abbrev: str = ""
    note: str = ""

    def tag_abbrev(self) -> str:
        if self.abbrev:
            return util.sanitize(self.abbrev)
        if self.key:
            return util.abbrev(self.key)
        return util.sanitize(self.param)


@dataclasses.dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    kind: str
    workloads: list[str]
    params: dict[str, str]
    extra_args: list[str]
    axes: list[Axis]
    description: str = ""
    bucket: str = ""
    path: Path | None = None

    def resolved(self) -> tuple[dict[str, str], list[str], model_registry.Model]:
        """Base ConfigMap params and flag list for this experiment's model.

        Precedence: model < experiment < (axis, later).
        Model identity is applied last so an experiment can't silently serve a
        different checkpoint than the one named in its `model:` field.
        """
        model = model_registry.load(self.model)
        params: dict[str, str] = dict(model.params)
        params.update(self.params)
        params.update(model.identity_params())

        extra_args = list(model.extra_args)
        extra_args += self.extra_args
        return params, extra_args, model


def display_label(value: Any) -> str:
    text = "" if value is None else str(value)
    return AUTO_LABEL if text == "" else text


def _format_extra_args(template: str, value: Any, model_name: str) -> str:
    """Expand an axis's extra_args template.

    Available fields: {value}, and the model's {spec_decode_method} / {hf_repo}
    so a speculative-decoding axis doesn't have to repeat the method name.
    """
    model = model_registry.load(model_name)
    context = {
        "value": value,
        "spec_decode_method": model.spec_decode_method or "",
        "hf_repo": model.hf_repo,
        "model": model.name,
    }
    try:
        return template.format(**context)
    except KeyError as exc:
        util.die(
            f"axis extra_args template refers to unknown field {exc}; "
            f"available: {', '.join(sorted(context))}"
        )
    except (IndexError, ValueError) as exc:
        util.die(f"axis extra_args template is malformed ({exc}): {template}")


def _parse_axis_value(
    raw: Any, axis_key: str | None, extra_args_template: str, model_name: str, where: str
) -> AxisValue:
    if isinstance(raw, dict):
        params = {k: str(v) for k, v in (raw.get("params") or {}).items()}
        value = raw.get("value")
        if axis_key and value is not None:
            params.setdefault(axis_key, str(value))
        label = str(raw.get("label") or display_label(value if value is not None else ""))
        if label == AUTO_LABEL and not params and not raw.get("extra_args"):
            util.die(f"{where}: a value mapping needs at least a label, params or extra_args")
        extra = str(raw.get("extra_args") or "")
        if extra_args_template and not extra:
            extra = _format_extra_args(extra_args_template, value, model_name)
        return AxisValue(label=label, params=params, extra_args=extra)

    params = {axis_key: str(raw)} if axis_key else {}
    extra = (
        _format_extra_args(extra_args_template, raw, model_name)
        if extra_args_template
        else ""
    )
    return AxisValue(label=display_label(raw), params=params, extra_args=extra)


def _parse_axis(raw: Any, index: int, model_name: str, where: str) -> Axis:
    if not isinstance(raw, dict):
        util.die(f"{where}: strategy.axes[{index}] must be a mapping")
    key = raw.get("key")
    template = str(raw.get("extra_args") or "")
    param = str(raw.get("param") or (key.lower() if key else ""))
    if not param:
        util.die(f"{where}: strategy.axes[{index}] needs a 'key' or a 'param' name")
    raw_values = raw.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        util.die(f"{where}: axis '{param}' needs a non-empty 'values' list")
    if not key and not template and not all(isinstance(v, dict) for v in raw_values):
        util.die(
            f"{where}: axis '{param}' has no 'key' and no 'extra_args' template, so "
            f"each value must be a mapping saying what it changes"
        )
    values = [
        _parse_axis_value(v, key, template, model_name, f"{where}: axis '{param}'")
        for v in raw_values
    ]
    return Axis(
        param=param,
        values=values,
        key=key,
        abbrev=str(raw.get("abbrev") or ""),
        note=str(raw.get("note") or "").strip(),
    )


def _files() -> list[Path]:
    if not settings.EXPERIMENTS_DIR.is_dir():
        return []
    return sorted(settings.EXPERIMENTS_DIR.rglob("*.yaml"))


def experiment_path(name: str) -> Path:
    """Resolve an experiment by path, by group-qualified name, or by bare name.

    Experiments live in subdirectories (`single-gpu/`, `multi-gpu/`) because a
    result measured on one GPU and a result measured on two answer different
    questions, and the spec is where that distinction is easiest to see. The bare
    name keeps working — `./bench run baseline` should not have to know which
    group a spec was filed under — and only an actual collision asks for more.
    """
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml"):
        # An explicit file is taken at its word, so a bad one names itself in the
        # error rather than being mangled into a lookup under experiments/.
        return candidate.resolve()

    qualified = settings.EXPERIMENTS_DIR / f"{name}.yaml"
    if qualified.exists():
        return qualified

    wanted = util.sanitize(Path(name).name)
    matches = [p for p in _files() if p.stem in (name, wanted)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        util.die(
            f"'{name}' matches {len(matches)} experiments; say which: "
            f"{', '.join(_display(p) for p in matches)}"
        )
    return qualified


def _display(path: Path) -> str:
    """How a spec is named on the command line: group-qualified, no extension."""
    try:
        return path.relative_to(settings.EXPERIMENTS_DIR).with_suffix("").as_posix()
    except ValueError:
        return path.stem


def group(path: Path | None) -> str:
    """The subdirectory an experiment is filed under, '-' at the top level."""
    if path is None:
        return "-"
    relative = _display(path)
    return relative.rsplit("/", 1)[0] if "/" in relative else "-"


def available() -> list[str]:
    return [_display(p) for p in _files()]


def load(name: str) -> Experiment:
    path = experiment_path(name)
    if not path.exists():
        known = ", ".join(available()) or "(none)"
        util.die(f"unknown experiment '{name}'. Available: {known}\n    (expected {path})")
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    where = str(path)

    model = raw.get("model")
    if not model:
        util.die(f"{where}: missing 'model' (one of: {', '.join(model_registry.available())})")

    strategy = raw.get("strategy") or {}
    kind = str(strategy.get("kind") or "single").lower()
    if kind not in KINDS:
        util.die(f"{where}: strategy.kind '{kind}' must be one of {', '.join(KINDS)}")

    params = {k: str(v) for k, v in (raw.get("params") or {}).items()}
    clashes = sorted(RESERVED_PARAM_KEYS & set(params))
    if clashes:
        util.die(
            f"{where}: {', '.join(clashes)} cannot be set under 'params'. "
            f"Use the 'extra_args' list for flags, and the model registry for identity."
        )

    axes = [
        _parse_axis(a, i, str(model), where)
        for i, a in enumerate(strategy.get("axes") or [])
    ]
    if kind in ("ofat", "grid") and not axes:
        util.die(f"{where}: strategy.kind '{kind}' needs at least one axis")
    if kind == "single" and axes:
        util.die(f"{where}: strategy.kind 'single' cannot have axes (use ofat or grid)")

    workloads = [str(w) for w in (raw.get("workloads") or ["rate", "concurrent"])]

    return Experiment(
        name=str(raw.get("name") or path.stem),
        model=str(model),
        kind=kind,
        workloads=workloads,
        params=params,
        extra_args=[str(a) for a in (raw.get("extra_args") or [])],
        axes=axes,
        description=str(raw.get("description") or "").strip(),
        bucket=str(raw.get("bucket") or "").strip(),
        path=path,
    )
