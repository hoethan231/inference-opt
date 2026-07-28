#!/usr/bin/env python3
"""
run_grid.py - full-factorial (grid search) sweep driven by a simple JSON file.

Where sweep/run_sweep.py varies one knob at a time (OFAT), this runs the full
Cartesian product of the values you list, so you can find the best *joint*
combination of parameters instead of the best independent value of each.

You describe the search in a small JSON config (default: config/grid.json):

    {
      "name": "grid",                     # tag prefix / experiment name
      "bench": "both",                    # rate | concurrent | both
      "base_overrides": {                 # applied to EVERY combo (held fixed)
        "MAX_MODEL_LEN": "131072"
      },
      "extra_args_append": "",            # appended to EXTRA_ARGS on every combo
      "grid": {                           # the axes to sweep (full factorial)
        "MAX_NUM_SEQS": ["64", "128", "256"],
        "MAX_NUM_BATCHED_TOKENS": ["2048", "4096"]
      }
    }

The keys are vLLM ConfigMap keys (see config/vllm-params.yaml). The example
above is 3 x 2 = 6 combinations. The run count is the product of the list
lengths, so add axes/values carefully -- it grows fast.

For each combination it reuses the exact same machinery as run_sweep.py
(render a ConfigMap -> `run.sh all` -> `run.sh pull`) but records to a
SEPARATE manifest (sweep/grid_manifest.jsonl) so grid runs and OFAT runs never
clobber each other. It is resumable via --resume.

Usage:
    ./sweep/run_grid.py --dry-run                 # show the plan + run count
    ./sweep/run_grid.py                           # run config/grid.json
    ./sweep/run_grid.py --config config/my.json   # a different search
    ./sweep/run_grid.py --bench rate              # faster (rate test only)
    ./sweep/run_grid.py --resume                  # continue after interruption
    ./sweep/run_grid.py --only mns-256            # only tags containing this

Then plot + rank:
    python3 sweep/plot_grid.py
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_CONFIG = REPO / "config" / "grid.json"
GRID_MANIFEST = HERE / "grid_manifest.jsonl"

sys.path.insert(0, str(HERE))
import run_sweep  # noqa: E402  (reuse its run_one/generate_configmap/etc.)
from grid import sanitize  # noqa: E402

# Short, human-guessable abbreviations for common knobs so grid tags stay
# under the k8s 63-char RFC-1123 limit. Anything not listed is abbreviated
# automatically from the first letter of each underscore-separated word.
KEY_ABBREV = {
    "MAX_NUM_SEQS": "mns",
    "MAX_NUM_BATCHED_TOKENS": "mnbt",
    "GPU_MEMORY_UTILIZATION": "gmu",
    "KV_CACHE_DTYPE": "kcd",
    "MAX_MODEL_LEN": "mml",
    "BLOCK_SIZE": "bs",
    "VLLM_ATTENTION_BACKEND": "ab",
    "TENSOR_PARALLEL_SIZE": "tp",
}


def abbrev(key):
    if key in KEY_ABBREV:
        return KEY_ABBREV[key]
    return "".join(part[0] for part in key.lower().split("_") if part) or "k"


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    grid = cfg.get("grid") or {}
    if not grid:
        sys.exit("config '{}' has no non-empty 'grid' object".format(path))
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            sys.exit("grid['{}'] must be a non-empty list of values".format(key))
    return cfg


def build_plan(cfg):
    """Expand the config into the full Cartesian product of grid points.

    Each plan item matches the schema run_sweep.run_one() expects, so we can
    reuse it verbatim. `value` is a compact human label of just the swept
    axes; `overrides` is the full ConfigMap patch (base_overrides + this combo).
    """
    name = sanitize(cfg.get("name") or "grid")
    base_overrides = cfg.get("base_overrides") or {}
    extra_args_append = (cfg.get("extra_args_append") or "").strip()
    grid = cfg["grid"]

    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    plan = []
    for idx, combo in enumerate(itertools.product(*value_lists)):
        combo_map = dict(zip(keys, combo))
        overrides = dict(base_overrides)
        overrides.update(combo_map)

        label = ", ".join("{}={}".format(k, v) for k, v in combo_map.items())
        short = "-".join(
            "{}-{}".format(abbrev(k), sanitize(v)) for k, v in combo_map.items()
        )
        tag = sanitize("{}-{:03d}-{}".format(name, idx, short))[:63].rstrip("-")

        item = dict(
            tag=tag,
            base_tag=tag,
            suffix="",
            param="grid",
            key=None,
            value=label,
            overrides=overrides,
        )
        if extra_args_append:
            item["extra_args_append"] = extra_args_append
        plan.append(item)
    return plan


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="grid search JSON config (default: config/grid.json)")
    ap.add_argument("--bench", default=None, choices=["rate", "concurrent", "both"],
                    help="override the config's 'bench' (default: from config, "
                    "else 'both')")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip combos already marked 'success' in grid_manifest.jsonl")
    ap.add_argument("--only", default=None,
                    help="only run combos whose tag contains this substring")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort on the first failed combo (default: continue)")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="retry a failed combo this many times before giving up "
                    "(default: 1)")
    ap.add_argument("--gpu-wait-timeout", type=int, default=180,
                    help="seconds to wait for a foreign workload to free the "
                    "shared GPU before deploying anyway (default: 180)")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit("config not found: {}".format(config_path))
    cfg = load_config(config_path)
    bench = args.bench or cfg.get("bench") or "both"

    plan = build_plan(cfg)
    if args.only:
        plan = [p for p in plan if args.only in p["tag"]]
        if not plan:
            sys.exit("no combos match --only '{}'".format(args.only))

    # Redirect run_sweep's manifest writer to the grid manifest so grid runs
    # stay separate from the OFAT sweep. (run_one() writes via the module-level
    # append_manifest(), which reads this global.)
    run_sweep.MANIFEST_PATH = GRID_MANIFEST

    prior_status = run_sweep.load_manifest_status() if args.resume else {}
    if args.resume:
        plan = [p for p in plan if prior_status.get(p["tag"]) != "success"]

    print("Grid search: {}".format(config_path))
    print("Axes:")
    for key, values in cfg["grid"].items():
        print("  - {:<26} {}".format(key, values))
    if cfg.get("base_overrides"):
        print("Held fixed (base_overrides): {}".format(cfg["base_overrides"]))
    print("\nPlanned combos ({}):".format(len(plan)))
    for item in plan:
        print("  - {:<40} {}".format(item["tag"], item["value"]))
    per = 28 if bench == "both" else (18 if bench == "concurrent" else 8)
    est = len(plan) * per
    print("\nEstimated wall-clock: ~{} min (~{:.1f} h) at bench={}".format(
        est, est / 60.0, bench))
    print("Manifest: {}".format(GRID_MANIFEST))

    if args.dry_run:
        return

    base_manifest = run_sweep.load_base_manifest()
    results = {"success": [], "failed": []}
    try:
        for item in plan:
            status = "failed"
            for attempt in range(args.max_retries + 1):
                if attempt > 0:
                    print("[{}] retrying (attempt {}/{})".format(
                        item["tag"], attempt + 1, args.max_retries + 1))
                status = run_sweep.run_one(
                    item, base_manifest, bench, args.gpu_wait_timeout,
                    attempt=attempt,
                )
                if status == "success":
                    break
            results[status if status in results else "failed"].append(item["tag"])
            if status != "success" and args.stop_on_error:
                print("\nStopping grid on first failure (--stop-on-error).")
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; rerun with --resume to continue.")
        return

    print("\n{}".format("=" * 78))
    print("Grid done: {} succeeded, {} failed".format(
        len(results["success"]), len(results["failed"])))
    if results["failed"]:
        print("Failed tags: {}".format(", ".join(results["failed"])))
    print("Next: python3 sweep/plot_grid.py")


if __name__ == "__main__":
    main()
