"""Read facts and failure causes out of a vLLM server log.

Two jobs, both previously done by a human squinting at sweep/logs/<tag>.log:

* `facts()` pulls out the numbers you need to interpret a result — a config with
  only a sliver of KV cache left is slow for a reason that isn't visible in the
  latency numbers alone.
* `classify_failure()` recognises the handful of ways a config is simply
  infeasible on this GPU (almost always memory), so a run can be abandoned in
  seconds instead of waiting out the rollout timeout.
"""

from __future__ import annotations

import re

# (fact name, pattern, converter). First match wins.
_FACT_PATTERNS: list[tuple[str, re.Pattern[str], type]] = [
    ("vllm_version", re.compile(r"vLLM API server version ([\w.+]+)"), str),
    ("vllm_version", re.compile(r"Initializing a V\d+ LLM engine \(v([\w.+]+)\)"), str),
    ("kv_cache_tokens", re.compile(r"GPU KV cache size: ([\d,]+) tokens"), int),
    ("kv_cache_gib", re.compile(r"Available KV cache memory: ([\d.]+) GiB"), float),
    ("gpu_blocks", re.compile(r"#\s*GPU blocks:\s*([\d,]+)"), int),
    (
        "max_concurrency",
        re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
        float,
    ),
    (
        "engine_init_s",
        re.compile(r"init engine .*?took ([\d.]+) seconds"),
        float,
    ),
    (
        "graph_capture_s",
        re.compile(r"Graph capturing finished in ([\d.]+) secs", re.IGNORECASE),
        float,
    ),
    ("torch_version", re.compile(r"PyTorch version:\s*([\w.+]+)"), str),
]

# Unambiguous "this config does not fit / cannot start" signals. Seeing one of
# these means we stop waiting immediately.
_FATAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("oom", re.compile(r"torch\.(?:cuda\.)?OutOfMemoryError")),
    ("oom", re.compile(r"CUDA out of memory")),
    ("oom", re.compile(r"No available memory for the cache blocks")),
    ("oom", re.compile(r"is larger than the available KV cache memory")),
    ("oom", re.compile(r"Free memory on device .* is less than desired GPU memory")),
    ("oom", re.compile(r"NCCL error.*out of memory", re.IGNORECASE)),
    ("startup", re.compile(r"EngineCore failed to start")),
    ("startup", re.compile(r"Engine core initialization failed")),
    ("startup", re.compile(r"ValueError: Unsupported")),
    ("startup", re.compile(r"does not support .* attention backend", re.IGNORECASE)),
    ("startup", re.compile(r"OSError: .*(?:not a valid model identifier|401 Client Error)")),
]

# Weaker evidence, used only to label a failure we already know happened.
_LABEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("oom", re.compile(r"out of memory", re.IGNORECASE)),
    ("startup", re.compile(r"^\s*(?:ValueError|RuntimeError|AssertionError|TypeError):", re.M)),
    ("startup", re.compile(r"Traceback \(most recent call last\)")),
]


def facts(log: str) -> dict[str, object]:
    """Engine numbers worth recording alongside a run's results."""
    found: dict[str, object] = {}
    for name, pattern, cast in _FACT_PATTERNS:
        if name in found:
            continue
        match = pattern.search(log)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        try:
            found[name] = cast(raw)
        except ValueError:
            continue
    return found


def _first_match(log: str, patterns: list[tuple[str, re.Pattern[str]]]) -> tuple[str, str] | None:
    best: tuple[int, str, str] | None = None
    for kind, pattern in patterns:
        match = pattern.search(log)
        if match and (best is None or match.start() < best[0]):
            line = log[match.start() : match.start() + 400].splitlines()[0].strip()
            best = (match.start(), kind, line)
    return (best[1], best[2]) if best else None


def classify_failure(log: str) -> tuple[str, str] | None:
    """(kind, evidence) for a log that shows a config which cannot start."""
    return _first_match(log, _FATAL_PATTERNS)


def label_failure(log: str) -> tuple[str, str]:
    """Best-effort reason for a crash we already detected another way."""
    fatal = classify_failure(log)
    if fatal:
        return fatal
    weak = _first_match(log, _LABEL_PATTERNS)
    if weak:
        return weak
    return ("crashed", "container exited without a recognisable error")


def tail(log: str, lines: int = 40) -> str:
    return "\n".join(log.splitlines()[-lines:])
