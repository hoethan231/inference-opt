"""The models/ registry: one file per servable model."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from . import settings, util


@dataclasses.dataclass(frozen=True)
class Model:
    name: str
    hf_repo: str
    cache_pvc: str
    params: dict[str, str]
    extra_args: list[str]
    revision: str | None = None
    image: str | None = None
    spec_decode_method: str | None = None
    note: str = ""
    path: Path | None = None

    @property
    def served_name(self) -> str:
        return self.name

    @property
    def cache_dir_name(self) -> str:
        """How the HF cache names this repo's directory under $HF_HOME/hub."""
        return "models--" + self.hf_repo.replace("/", "--")

    def identity_params(self) -> dict[str, str]:
        """ConfigMap keys that are the model, not a tunable."""
        return {"MODEL_NAME": self.hf_repo, "SERVED_MODEL_NAME": self.served_name}


def _model_path(name: str) -> Path:
    # Not sanitized: model names legitimately contain dots (laguna-s-2.1), and
    # they double as --served-model-name, so the file name is the name.
    if "/" in name or name.startswith("."):
        util.die(f"invalid model name '{name}'")
    return settings.MODELS_DIR / f"{name}.yaml"


def available() -> list[str]:
    if not settings.MODELS_DIR.is_dir():
        return []
    names: list[str] = []
    for path in sorted(settings.MODELS_DIR.glob("*.yaml")):
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        if bool(raw.get("abstract", False)):
            continue
        names.append(path.stem)
    return names


def _raw_model(name: str) -> tuple[Path, dict[str, Any]]:
    path = _model_path(name)
    if not path.exists():
        known = ", ".join(available()) or "(none)"
        util.die(f"unknown model '{name}'. Available: {known}\n    (expected {path})")
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return path, raw


def _resolve_raw(name: str, stack: tuple[str, ...] = ()) -> tuple[Path, dict[str, Any]]:
    if name in stack:
        chain = " -> ".join((*stack, name))
        util.die(f"model inheritance cycle detected: {chain}")
    path, raw = _raw_model(name)

    parent_name = str(raw.get("extends") or "").strip()
    if not parent_name:
        merged = dict(raw)
        merged.pop("extends", None)
        merged.pop("abstract", None)
        return path, merged

    parent_path, parent = _resolve_raw(parent_name, (*stack, name))
    merged: dict[str, Any] = dict(parent)

    parent_params = {k: str(v) for k, v in (parent.get("params") or {}).items()}
    child_params = {k: str(v) for k, v in (raw.get("params") or {}).items()}
    merged["params"] = {**parent_params, **child_params}

    parent_args = [str(a) for a in (parent.get("extra_args") or [])]
    if "extra_args" in raw:
        child_args = [str(a) for a in (raw.get("extra_args") or [])]
    else:
        child_args = parent_args
    remove_args = {str(a) for a in (raw.get("extra_args_remove") or [])}
    if remove_args:
        child_args = [a for a in child_args if a not in remove_args]
    append_args = [str(a) for a in (raw.get("extra_args_append") or [])]
    if append_args:
        child_args = [*child_args, *append_args]
    merged["extra_args"] = child_args

    for key, value in raw.items():
        if key in {"extends", "abstract", "params", "extra_args", "extra_args_append", "extra_args_remove"}:
            continue
        merged[key] = value
    merged["name"] = str(raw.get("name") or path.stem)
    return path, merged


def load(name: str) -> Model:
    path, source = _raw_model(name)
    if bool(source.get("abstract", False)):
        util.die(f"model '{name}' is abstract and cannot be used directly")

    path, raw = _resolve_raw(name)

    for required in ("hf_repo", "cache_pvc"):
        if not raw.get(required):
            util.die(f"{path}: missing required key '{required}'")

    params = {k: str(v) for k, v in (raw.get("params") or {}).items()}
    extra_args = [str(a) for a in (raw.get("extra_args") or [])]
    revision = str(raw.get("revision") or "").strip() or None
    if not revision:
        for arg in extra_args:
            if arg.startswith("--revision="):
                value = arg.split("=", 1)[1].strip()
                if value:
                    revision = value
                    break
    return Model(
        name=str(raw.get("name") or path.stem),
        hf_repo=str(raw["hf_repo"]),
        cache_pvc=str(raw["cache_pvc"]),
        params=params,
        extra_args=extra_args,
        revision=revision,
        image=raw.get("image"),
        spec_decode_method=raw.get("spec_decode_method"),
        note=str(raw.get("note") or ""),
        path=path,
    )
