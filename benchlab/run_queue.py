"""Persistent queue for `bench run` requests."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any

from . import settings, util


@dataclasses.dataclass
class QueueItem:
    id: str
    enqueued_at: str
    experiment: str
    options: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "QueueItem":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclasses.dataclass
class ActiveRun:
    pid: int
    host: str
    started_at: str
    experiment: str


def _safe_read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _safe_write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _is_live_pid(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def enqueue(experiment: str, options: dict[str, Any]) -> tuple[QueueItem, int]:
    item = QueueItem(
        id=uuid.uuid4().hex[:8],
        enqueued_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        experiment=experiment,
        options=options,
    )
    lines = _safe_read_lines(settings.QUEUE_PATH)
    lines.append(json.dumps(item.to_json(), sort_keys=True))
    _safe_write_lines(settings.QUEUE_PATH, lines)
    return item, len(lines)


def list_items() -> list[QueueItem]:
    items: list[QueueItem] = []
    for lineno, line in enumerate(_safe_read_lines(settings.QUEUE_PATH), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(QueueItem.from_json(json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            util.warn(f"{settings.QUEUE_PATH}:{lineno}: skipping unreadable queue item")
    return items


def remove_at(position: int) -> QueueItem:
    if position < 1:
        util.die("queue position must be >= 1")
    items = list_items()
    if position > len(items):
        util.die(f"queue position {position} is out of range (1..{len(items) or 0})")
    removed = items[position - 1]
    lines = [json.dumps(item.to_json(), sort_keys=True) for i, item in enumerate(items) if i != position - 1]
    _safe_write_lines(settings.QUEUE_PATH, lines)
    return removed


def pop_next() -> QueueItem | None:
    lines = _safe_read_lines(settings.QUEUE_PATH)
    if not lines:
        return None
    rest = lines
    while rest:
        first = rest[0].strip()
        rest = rest[1:]
        if not first:
            continue
        try:
            item = QueueItem.from_json(json.loads(first))
        except (json.JSONDecodeError, TypeError):
            util.warn(f"{settings.QUEUE_PATH}: dropping unreadable first queue item")
            continue
        _safe_write_lines(settings.QUEUE_PATH, [line for line in rest if line.strip()])
        return item
    _safe_write_lines(settings.QUEUE_PATH, [])
    return None


def active_run() -> ActiveRun | None:
    path = settings.QUEUE_ACTIVE_PATH
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        active = ActiveRun(
            pid=int(payload.get("pid") or 0),
            host=str(payload.get("host") or ""),
            started_at=str(payload.get("started_at") or ""),
            experiment=str(payload.get("experiment") or ""),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if active.pid <= 0:
        return None
    if active.host and active.host != socket.gethostname():
        return active
    if not _is_live_pid(active.pid):
        clear_active()
        return None
    return active


def try_acquire_active(experiment: str) -> bool:
    existing = active_run()
    if existing and existing.pid != os.getpid():
        return False
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
    }
    settings.QUEUE_ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.QUEUE_ACTIVE_PATH.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return True


def clear_active() -> None:
    try:
        if settings.QUEUE_ACTIVE_PATH.exists():
            settings.QUEUE_ACTIVE_PATH.unlink()
    except OSError:
        pass
