"""The run store: runs/index.jsonl plus one directory of artefacts per run.

Replaces sweep/manifest.jsonl *and* sweep/grid_manifest.jsonl. Those were split
because run_grid.py reused run_sweep.py's writer and had to monkeypatch its
module-level path to avoid clobbering; the two files then needed a magic
`param == "grid"` field to tell apart, which is why there were two plotting
scripts. One append-only index with an explicit `experiment` and `model` on
every record removes the whole problem.

Append-only: an entry is never rewritten, so a killed sweep can always be
resumed and a run's history (including retries) stays visible. Readers take the
last entry per tag.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
from typing import Any, Iterator

from . import parallelism, settings, util

SUCCESS = "success"
FAILED = "failed"
INFEASIBLE = "infeasible"  # the config cannot start on this GPU (usually memory)
INTERRUPTED = "interrupted"

TERMINAL_OK = {SUCCESS}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@dataclasses.dataclass
class RunRecord:
    """One attempt at one config. Mirrors the JSONL schema."""

    tag: str
    experiment: str
    strategy: str
    model: str
    hf_repo: str = ""
    param: str = ""
    value: str = ""
    label: str = ""
    suffix: str = ""
    overrides: dict[str, str] = dataclasses.field(default_factory=dict)
    params: dict[str, str] = dataclasses.field(default_factory=dict)
    extra_args: str = ""
    extra_args_append: str = ""
    workloads: list[str] = dataclasses.field(default_factory=list)
    status: str = FAILED
    attempt: int = 0
    failure_kind: str = ""
    failure_detail: str = ""
    engine: dict[str, Any] = dataclasses.field(default_factory=dict)
    smoke: str = ""
    results: dict[str, str] = dataclasses.field(default_factory=dict)
    elapsed_s: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    run_dir: str = ""
    note: str = ""

    @property
    def topology(self) -> parallelism.Topology:
        """The GPU shape this attempt ran on, derived from its own config.

        Derived rather than stored so it cannot disagree with the params that were
        actually applied, and so runs recorded before multi-GPU support read
        correctly: they have no parallelism keys, and they were all single-GPU.
        """
        return parallelism.of(self.params)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "RunRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def run_dir(tag: str) -> Path:
    return settings.RUNS_DIR / tag


def append(record: RunRecord, *, index_path: Path = settings.INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")


def iter_records(index_path: Path = settings.INDEX_PATH) -> Iterator[RunRecord]:
    if not index_path.exists():
        return
    with open(index_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield RunRecord.from_json(json.loads(line))
            except (json.JSONDecodeError, TypeError) as exc:
                util.warn(f"{index_path}:{lineno}: skipping unreadable entry ({exc})")


def latest_by_tag(index_path: Path = settings.INDEX_PATH) -> dict[str, RunRecord]:
    """Last entry wins, which is what makes retries and --resume behave."""
    records: dict[str, RunRecord] = {}
    for record in iter_records(index_path):
        records[record.tag] = record
    return records


def successful_tags(index_path: Path = settings.INDEX_PATH) -> set[str]:
    return {
        tag for tag, rec in latest_by_tag(index_path).items() if rec.status in TERMINAL_OK
    }


def result_files(record: RunRecord) -> dict[str, Path]:
    """Workload -> local result JSON, for records whose files still exist."""
    found = {}
    for workload, relative in (record.results or {}).items():
        path = settings.REPO / relative if not Path(relative).is_absolute() else Path(relative)
        if path.exists():
            found[workload] = path
    return found


def experiments(index_path: Path = settings.INDEX_PATH) -> list[str]:
    return sorted({r.experiment for r in latest_by_tag(index_path).values() if r.experiment})


def select(
    *,
    experiment: str | None = None,
    model: str | None = None,
    tags: list[str] | None = None,
    suffix: str | None = None,
    status: set[str] | None = None,
    index_path: Path = settings.INDEX_PATH,
) -> list[RunRecord]:
    """Records matching every supplied filter, in index order."""
    chosen = []
    for record in latest_by_tag(index_path).values():
        if experiment and record.experiment != experiment:
            continue
        if model and record.model != model:
            continue
        if tags and record.tag not in tags:
            continue
        if suffix is not None and (record.suffix or "") != suffix:
            continue
        if status and record.status not in status:
            continue
        chosen.append(record)
    return sorted(chosen, key=lambda r: (r.experiment, r.tag))


def resolve_tag(needle: str, index_path: Path = settings.INDEX_PATH) -> RunRecord:
    """Exact tag, else a unique substring match — handy for long grid tags."""
    records = latest_by_tag(index_path)
    if needle in records:
        return records[needle]
    matches = [rec for tag, rec in records.items() if needle in tag]
    if len(matches) == 1:
        util.step(f"'{needle}' matched run {matches[0].tag}")
        return matches[0]
    if not matches:
        util.die(f"no run matching '{needle}' in {index_path}")
    util.die(
        f"'{needle}' matches {len(matches)} runs, be more specific: "
        f"{', '.join(sorted(m.tag for m in matches)[:8])}"
    )
