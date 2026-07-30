"""One-shot import of the pre-refactor results into the new run store.

Reads the two old manifests (sweep/manifest.jsonl for the OFAT sweep,
sweep/grid_manifest.jsonl for grid searches) plus results/outputs/, and rewrites
them as index entries and per-run directories. Result files are *moved*, not
copied — there are gigabytes of them.

Old runs keep their original tags, so `bench compare` works across the refactor.
They land in experiments named `legacy-*` so they don't blend into new sweeps
unless you ask for them.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from . import model_registry, settings, store, util
from .store import RunRecord

LEGACY_RESULTS = settings.REPO / "results" / "outputs"
LEGACY_PLOTS = settings.REPO / "results" / "plots"
LEGACY_SWEEP = settings.REPO / "sweep"
LEGACY_MANIFEST = LEGACY_SWEEP / "manifest.jsonl"
LEGACY_GRID_MANIFEST = LEGACY_SWEEP / "grid_manifest.jsonl"
LEGACY_LOGS = LEGACY_SWEEP / "logs"
ARCHIVE = settings.REPO / "archive"

# The base params are read to reconstruct what a legacy run was serving. Once
# this import has happened the file itself is archived, so look in both places.
LEGACY_BASE_PARAMS_CANDIDATES = (
    settings.REPO / "config" / "vllm-params.yaml",
    ARCHIVE / "legacy-config" / "vllm-params.yaml",
)

# Where the old per-run ConfigMaps live, before and after they were archived.
LEGACY_CONFIG_DIRS = (settings.REPO / "config", ARCHIVE / "legacy-config")

# vllm-rate-<tag>.json / vllm-users-<tag>.html
FILE_RE = re.compile(r"^vllm-(rate|users)-(.+)\.(json|html)$")
WORKLOAD_FOR_KIND = {"rate": "rate", "users": "concurrent"}
GRID_TAG_RE = re.compile(r"^(?P<prefix>.+?)-\d{3}-")

BENCH_TO_WORKLOADS = {
    "rate": ["rate"],
    "concurrent": ["concurrent"],
    "both": ["rate", "concurrent"],
}


def _legacy_base_params() -> dict[str, str]:
    for path in LEGACY_BASE_PARAMS_CANDIDATES:
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            return {k: str(v) for k, v in (raw.get("data") or {}).items()}
    return {}


def _configs_by_tag() -> dict[str, tuple[Path, dict[str, str]]]:
    """Every archived per-run ConfigMap, keyed by the tag it was applied for.

    The old drivers stamped `sweep-tag` on each rendered ConfigMap, which makes
    these files the only surviving record of what a run *actually* served.
    """
    found: dict[str, tuple[Path, dict[str, str]]] = {}
    for directory in LEGACY_CONFIG_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            if raw.get("kind") != "ConfigMap":
                continue
            tag = ((raw.get("metadata") or {}).get("labels") or {}).get("sweep-tag")
            data = raw.get("data") or {}
            if tag and data:
                found[str(tag)] = (path, {k: str(v) for k, v in data.items()})
    return found


def repair_configs(*, dry_run: bool = False) -> int:
    """Give imported runs their real config back, where a ConfigMap survives.

    A run made outside the sweep driver has no config in either manifest, so the
    import could only fall back to the base params file — which has since moved
    on. Reports and `bench compare` flag such a config as reconstructed; this
    replaces it with the real one and appends the correction, leaving the
    original entry in place as history.
    """
    configs = _configs_by_tag()
    if not configs:
        return 0
    base_params = _legacy_base_params()
    repaired = 0
    for tag, record in sorted(store.latest_by_tag().items()):
        if record.overrides or "recovered" in (record.note or ""):
            continue  # its config is already known
        found = configs.get(tag)
        if not found:
            continue
        path, params = found
        try:
            source = path.relative_to(settings.REPO)
        except ValueError:
            source = path
        record.params = params
        record.extra_args = params.get("EXTRA_ARGS", "")
        record.overrides = {
            key: value
            for key, value in params.items()
            if base_params.get(key) != value
        }
        record.note = f"config recovered from {source}"
        util.step(f"{tag}: config recovered from {source}")
        if not dry_run:
            store.append(record)
        repaired += 1
    return repaired


def _model_for(overrides: dict[str, str], default: str) -> tuple[str, str]:
    hf_repo = overrides.get("MODEL_NAME", "")
    if hf_repo:
        for name in model_registry.available():
            if model_registry.load(name).hf_repo == hf_repo:
                return name, hf_repo
        return util.sanitize(hf_repo.split("/")[-1]), hf_repo
    try:
        model = model_registry.load(default)
        return model.name, model.hf_repo
    except SystemExit:
        return default, ""


def _read_manifest(path: Path, strategy: str) -> dict[str, dict[str, Any]]:
    """tag -> last legacy record, tagged with the strategy it came from."""
    found: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return found
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag = record.get("tag")
        if not tag:
            continue
        record["_strategy"] = strategy
        found[tag] = record
    return found


def _experiment_name(tag: str, strategy: str) -> str:
    if strategy == "grid":
        match = GRID_TAG_RE.match(tag)
        return f"legacy-{match.group('prefix')}" if match else "legacy-grid"
    return "legacy-ofat"


def _collect_files() -> tuple[dict[str, dict[str, list[Path]]], list[Path]]:
    """(tag -> workload -> files, unrecognised files)."""
    grouped: dict[str, dict[str, list[Path]]] = {}
    other: list[Path] = []
    if not LEGACY_RESULTS.is_dir():
        return grouped, other
    for path in sorted(LEGACY_RESULTS.iterdir()):
        if not path.is_file():
            continue
        match = FILE_RE.match(path.name)
        if not match:
            other.append(path)
            continue
        kind, tag, _ = match.groups()
        workload = WORKLOAD_FOR_KIND[kind]
        grouped.setdefault(tag, {}).setdefault(workload, []).append(path)
    return grouped, other


def _build_record(
    tag: str,
    legacy: dict[str, Any] | None,
    files: dict[str, list[Path]],
    base_params: dict[str, str],
    default_model: str,
) -> RunRecord:
    legacy = legacy or {}
    strategy = legacy.get("_strategy") or ("grid" if GRID_TAG_RE.match(tag) else "ofat")
    overrides = {k: str(v) for k, v in (legacy.get("overrides") or {}).items()}
    model_name, hf_repo = _model_for(overrides, default_model)

    params = dict(base_params)
    params.update(overrides)
    append = str(legacy.get("extra_args_append") or "")
    if append:
        params["EXTRA_ARGS"] = f"{params.get('EXTRA_ARGS', '').strip()} {append}".strip()

    param = str(legacy.get("param") or ("grid" if strategy == "grid" else "unknown"))
    raw_value = legacy.get("value")
    if param == "baseline" or raw_value in (None, ""):
        value = "baseline" if param == "baseline" else (raw_value or "auto")
    else:
        value = str(raw_value)

    key = legacy.get("key")
    if param == "grid":
        label = str(raw_value or ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())))
        value = label
    elif key:
        label = f"{key}={value}"
    else:
        label = f"{param}={value}" if param not in ("baseline",) else "baseline"

    workloads = BENCH_TO_WORKLOADS.get(str(legacy.get("bench") or "both"), ["rate", "concurrent"])
    if files:
        workloads = sorted(set(workloads) | set(files))

    experiment = (
        _experiment_name(tag, strategy) if legacy else "legacy-untracked"
    )

    return RunRecord(
        tag=tag,
        experiment=experiment,
        strategy=strategy,
        model=model_name,
        hf_repo=hf_repo,
        param=param,
        value=value,
        label=label,
        suffix=str(legacy.get("suffix") or ""),
        overrides=overrides,
        params=params,
        extra_args=params.get("EXTRA_ARGS", ""),
        extra_args_append=append,
        workloads=workloads,
        status=str(legacy.get("status") or ("success" if files else store.FAILED)),
        elapsed_s=float(legacy.get("elapsed_s") or 0),
        started_at=str(legacy.get("started_at") or ""),
        finished_at=str(legacy.get("finished_at") or ""),
        run_dir=str(store.run_dir(tag).relative_to(settings.REPO)),
        note="imported from the pre-refactor layout; params reconstructed from "
        "the old config/vllm-params.yaml plus the recorded overrides",
    )


def run(*, dry_run: bool = False) -> int:
    base_params = _legacy_base_params()
    legacy_records = _read_manifest(LEGACY_MANIFEST, "ofat")
    legacy_records.update(_read_manifest(LEGACY_GRID_MANIFEST, "grid"))
    files_by_tag, unrecognised = _collect_files()

    tags = sorted(set(legacy_records) | set(files_by_tag))
    if not tags:
        util.log("nothing to import: no legacy manifests or result files found")
        repaired = repair_configs(dry_run=dry_run)
        if repaired:
            util.ok(f"recovered the real config for {repaired} imported run(s)")
        return 0

    already = set(store.latest_by_tag())
    default_model = "nemotron-3-super-120b"

    imported, skipped, empty = 0, 0, 0
    for tag in tags:
        if tag in already:
            skipped += 1
            continue
        files = files_by_tag.get(tag, {})
        record = _build_record(tag, legacy_records.get(tag), files, base_params, default_model)
        target = store.run_dir(tag)
        results: dict[str, str] = {}
        for workload, paths in sorted(files.items()):
            for path in paths:
                extension = path.suffix.lstrip(".")
                destination = target / f"{workload}.{extension}"
                if not dry_run:
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(destination))
                if extension == "json":
                    results[workload] = str(destination.relative_to(settings.REPO))
        record.results = results
        if not results:
            empty += 1
        log_source = LEGACY_LOGS / f"{tag}.log"
        if log_source.exists() and not dry_run:
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(log_source), str(target / "legacy-run.log"))
        if not dry_run:
            store.append(record)
        imported += 1

    repaired = repair_configs(dry_run=dry_run)

    util.rule("migration")
    print(
        util.table(
            ["outcome", "runs"],
            [
                ["imported", imported],
                ["already in the index (skipped)", skipped],
                ["imported without result files", empty],
                ["config recovered from an archived ConfigMap", repaired],
                ["unrecognised files to archive", len(unrecognised)],
            ],
        )
    )

    if dry_run:
        util.log("dry run: nothing was moved or written")
        return 0

    _archive(unrecognised)
    util.ok(f"index: {settings.INDEX_PATH}")
    util.log("check it over with: ./bench doctor")
    return 0


def _archive(unrecognised: list[Path]) -> None:
    """Park everything the new tool doesn't read, rather than deleting it."""
    if unrecognised:
        destination = ARCHIVE / "legacy-outputs"
        destination.mkdir(parents=True, exist_ok=True)
        for path in unrecognised:
            shutil.move(str(path), str(destination / path.name))
        util.step(f"archived {len(unrecognised)} unrecognised result file(s) -> {destination}")

    if LEGACY_PLOTS.is_dir():
        destination = ARCHIVE / "plots"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(LEGACY_PLOTS), str(destination))
        util.step(f"archived the old plots -> {destination}")

    manifests = ARCHIVE / "legacy-manifests"
    for path in (LEGACY_MANIFEST, LEGACY_GRID_MANIFEST):
        if path.exists():
            manifests.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(manifests / path.name))
            util.step(f"archived {path.name} -> {manifests}")

    if LEGACY_LOGS.is_dir():
        remaining = list(LEGACY_LOGS.iterdir())
        if remaining:
            destination = ARCHIVE / "legacy-logs"
            destination.mkdir(parents=True, exist_ok=True)
            for path in remaining:
                shutil.move(str(path), str(destination / path.name))
            util.step(f"archived {len(remaining)} leftover log(s) -> {destination}")

    # Drop the now-empty legacy directories.
    for path in (LEGACY_RESULTS, LEGACY_LOGS, LEGACY_RESULTS.parent):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
