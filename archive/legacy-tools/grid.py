#!/usr/bin/env python3
"""
grid.py - defines the one-factor-at-a-time (OFAT) hyperparameter sweep.

Every entry in SWEEP varies exactly one vLLM knob while holding everything
else at BASELINE. That keeps the run count linear (sum of |values| - 1)
instead of a full-factorial product, per the sweep design.

Run standalone to print the plan:
    python3 sweep/grid.py
    python3 sweep/grid.py --json
"""
import argparse
import json
import re

# Baseline = the values currently shipped in config/vllm-params.yaml. Every
# swept run starts here and changes exactly one key.
BASELINE = {
    "GPU_MEMORY_UTILIZATION": "0.9",
    "MAX_MODEL_LEN": "262144",
    "KV_CACHE_DTYPE": "fp8",
    "MAX_NUM_SEQS": "128",
    "MAX_NUM_BATCHED_TOKENS": "2048",
    "BLOCK_SIZE": "16",
    "VLLM_ATTENTION_BACKEND": "",  # "" = vLLM auto-selects a backend
}

# Human label for VLLM_ATTENTION_BACKEND="" in tags/CSVs, since an empty
# string makes a bad tag/column value.
AUTO_LABEL = "auto"

# One entry per swept parameter: which ConfigMap key it maps to, and the
# candidate values to try (baseline's own value is skipped automatically
# since the "baseline" run already covers it).
SWEEP = [
    dict(
        param="kv_cache_dtype",
        key="KV_CACHE_DTYPE",
        values=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
        note="KV-cache storage precision",
    ),
    dict(
        param="max_num_seqs",
        key="MAX_NUM_SEQS",
        values=["32", "64", "128", "256"],
        note="max concurrent running sequences (scheduler 'bouncer')",
    ),
    dict(
        param="max_num_batched_tokens",
        key="MAX_NUM_BATCHED_TOKENS",
        values=["512", "1024", "2048", "4096", "8192"],
        note="prefill chunk size (scheduler 'chewer')",
    ),
    dict(
        param="attention_backend",
        key="VLLM_ATTENTION_BACKEND",
        values=["", "FLASHINFER", "TRITON_ATTN"],
        note="VLLM_ATTENTION_BACKEND env var; XFORMERS omitted (V0-only, "
        "not supported on the V1 engine used here)",
    ),
    dict(
        param="gpu_memory_utilization",
        key="GPU_MEMORY_UTILIZATION",
        values=["0.80", "0.85", "0.9", "0.95"],
        note="fraction of VRAM reserved for weights + KV cache",
    ),
    dict(
        param="block_size",
        key="BLOCK_SIZE",
        values=["16", "32"],
        note="KV-cache block size (tokens/block)",
    ),
    dict(
        param="max_model_len",
        key="MAX_MODEL_LEN",
        values=["8192", "32768", "131072", "262144"],
        note="max context window; smaller frees more KV-cache blocks",
    ),
]

SPEC_DECODE_BASELINE_TOKENS = 0
SPEC_DECODE_TOKENS = [0, 3, 4, 5]

SPEC_DECODE = [
    dict(
        param="spec_decode_mtp",
        method="nemotron_h_mtp",
        note="native Multi-Token Prediction head (Nemotron-H MTP); no draft "
        "model required",
    ),
]


def spec_config_arg(method: str, num_speculative_tokens: int) -> str:
    """Build the shell-safe EXTRA_ARGS fragment for one spec-decode run.

    ``--max-cudagraph-capture-size=128`` is included because enabling MTP on top
    will run an OOM error
    """
    blob = json.dumps(
        {"method": method, "num_speculative_tokens": num_speculative_tokens},
        separators=(",", ":"),
    )
    return "--speculative-config={} --max-cudagraph-capture-size=128".format(blob)


def sanitize(value: str) -> str:
    """RFC-1123-ish label: lowercase alnum + dashes, matches run.sh's sanitize()."""
    s = str(value).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "default"


def display_value(key: str, value: str) -> str:
    if key == "VLLM_ATTENTION_BACKEND" and value == "":
        return AUTO_LABEL
    return value


def build_plan(suffix=None):
    """Returns the ordered list of sweep points: baseline + one per (param, value).

    If `suffix` is given, it is sanitized and appended to every tag
    (e.g. tag "gpu-memory-utilization-0-80" -> "gpu-memory-utilization-0-80-<suffix>").
    k8s Job names / ConfigMap labels are RFC-1123 (no underscores), so the
    separator is a dash even though you might type it as `_<suffix>`. The raw
    suffix is also stored on each plan item so tooling can label runs by it.
    """
    suffix_label = suffix or ""
    suffix_tag = sanitize(suffix) if suffix else ""

    def with_suffix(tag):
        return "{}-{}".format(tag, suffix_tag) if suffix_tag else tag

    plan = [
        dict(tag=with_suffix("baseline"), base_tag="baseline", suffix=suffix_label,
             param="baseline", key=None, value=None, overrides={})
    ]
    for sweep in SWEEP:
        key = sweep["key"]
        baseline_value = BASELINE[key]
        for value in sweep["values"]:
            if value == baseline_value:
                continue  # already covered by the baseline run
            base_tag = f"{sanitize(sweep['param'])}-{sanitize(display_value(key, value))}"
            plan.append(
                dict(
                    tag=with_suffix(base_tag),
                    base_tag=base_tag,
                    suffix=suffix_label,
                    param=sweep["param"],
                    key=key,
                    value=value,
                    overrides={key: value},
                )
            )
    for spec in SPEC_DECODE:
        for tokens in SPEC_DECODE_TOKENS:
            if tokens == SPEC_DECODE_BASELINE_TOKENS:
                continue 
            base_tag = "{}-{}".format(sanitize(spec["param"]), tokens)
            plan.append(
                dict(
                    tag=with_suffix(base_tag),
                    base_tag=base_tag,
                    suffix=suffix_label,
                    param=spec["param"],
                    key=None,
                    value=str(tokens),
                    overrides={},
                    extra_args_append=spec_config_arg(spec["method"], tokens),
                )
            )
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    args = ap.parse_args()

    plan = build_plan()
    if args.json:
        print(json.dumps(plan, indent=2))
        return

    print(f"{'tag':<32} {'param':<24} {'value':<12} overrides")
    print("-" * 90)
    for item in plan:
        if item["key"]:
            val = display_value(item["key"], item["value"])
            overrides = str(item["overrides"])
        elif item.get("extra_args_append"):
            val = item["value"]
            overrides = "EXTRA_ARGS += {}".format(item["extra_args_append"])
        else:
            val = "(baseline)"
            overrides = str(item["overrides"])
        print(f"{item['tag']:<32} {item['param']:<24} {val:<12} {overrides}")
    print("-" * 90)
    print(f"{len(plan)} configs total (1 baseline + {len(plan) - 1} variants)")


if __name__ == "__main__":
    main()
