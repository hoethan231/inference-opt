"""Small shared helpers: logging, name sanitizing, table rendering."""

from __future__ import annotations

import hashlib
import re
import sys
from typing import Any, Iterable, NoReturn, Sequence

BLUE = "\033[1;34m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
GREEN = "\033[1;32m"
DIM = "\033[2m"
RESET = "\033[0m"

_ISATTY = sys.stdout.isatty()
_ISATTY_ERR = sys.stderr.isatty()


def _paint(color: str, text: str) -> str:
    return f"{color}{text}{RESET}" if _ISATTY else text


def _paint_err(color: str, text: str) -> str:
    return f"{color}{text}{RESET}" if _ISATTY_ERR else text


def log(msg: str) -> None:
    print(f"{_paint(BLUE, '==>')} {msg}", flush=True)


def step(msg: str) -> None:
    print(f"{_paint(DIM, '  ·')} {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"{_paint(GREEN, '  ✓')} {msg}", flush=True)


def warn(msg: str) -> None:
    # Warnings belong on stderr, but stdout is block-buffered when piped: without
    # this flush they would surface far from the output they refer to.
    sys.stdout.flush()
    print(f"{_paint_err(YELLOW, '!! ')} {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> NoReturn:
    sys.stdout.flush()
    raise SystemExit(f"{_paint_err(RED, 'xx ')} {msg}")


def rule(title: str = "", width: int = 78) -> None:
    if not title:
        print("=" * width)
        return
    print("=" * width)
    print(title)
    print("=" * width)


def sanitize(value: Any) -> str:
    """Lowercase alphanumerics and dashes: an RFC-1123-safe label fragment."""
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower())
    return re.sub(r"-{2,}", "-", text).strip("-") or "default"


def abbrev(key: str) -> str:
    """Short form of a ConfigMap key, for keeping grid tags under the k8s limit."""
    known = {
        "MAX_NUM_SEQS": "mns",
        "MAX_NUM_BATCHED_TOKENS": "mnbt",
        "GPU_MEMORY_UTILIZATION": "gmu",
        "KV_CACHE_DTYPE": "kcd",
        "MAX_MODEL_LEN": "mml",
        "BLOCK_SIZE": "bs",
        "VLLM_ATTENTION_BACKEND": "ab",
        "TENSOR_PARALLEL_SIZE": "tp",
        "PIPELINE_PARALLEL_SIZE": "pp",
        "DATA_PARALLEL_SIZE": "dp",
        "EXPERT_PARALLEL": "ep",
    }
    if key in known:
        return known[key]
    return "".join(part[0] for part in key.lower().split("_") if part) or "k"


def fit_name(name: str, limit: int) -> str:
    """Truncate a tag to `limit`, keeping it unique via a short content hash."""
    if len(name) <= limit:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:5]
    return f"{name[: limit - 6].rstrip('-')}-{digest}"


def human_duration(seconds: float | None) -> str:
    if not seconds:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def fmt(value: Any, digits: int = 1) -> str:
    """Format a metric for a table cell, tolerating None/NaN."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        return f"{value:,.{digits}f}"
    return str(value)


def elide(text: str, width: int) -> str:
    """Shorten from the middle: both ends of a flag carry the meaning."""
    if len(text) <= width:
        return text
    keep = (width - 1) // 2
    return f"{text[:keep]}…{text[-keep:]}"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], indent: str = "") -> str:
    """Left-aligned fixed-width text table (we don't write CSVs anymore)."""
    body = [[("-" if c is None else str(c)) for c in row] for row in rows]
    if not body:
        return f"{indent}(no rows)"
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    lines = [
        indent + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
        indent + "  ".join("-" * w for w in widths),
    ]
    for row in body:
        lines.append(
            indent + "  ".join(
                cell.ljust(widths[i]) for i, cell in enumerate(row) if i < len(widths)
            ).rstrip()
        )
    return "\n".join(lines)
