#!/usr/bin/env python3
"""
run_sweep.py - schedules the OFAT grid (sweep/grid.py) one config at a time.

For each config it:
  1. renders a per-config ConfigMap YAML (baseline + one override) into
     config/generated/
  2. runs `PARAMS=<generated file> ./run.sh all <bench> <tag>`
     (deploy the server with that config, wait Ready, run benchmark(s))
  3. runs `./run.sh pull <tag>` to copy results back immediately
  4. appends one line to sweep/manifest.jsonl recording what ran and whether
     it succeeded

It is resumable: rerun with --resume and configs already marked "success" in
the manifest are skipped. Use --only to (re)run a subset by tag substring.

Usage:
    ./sweep/run_sweep.py --dry-run
    ./sweep/run_sweep.py                       # full sweep, bench=both
    ./sweep/run_sweep.py --bench rate           # faster sweep, rate test only
    ./sweep/run_sweep.py --resume               # continue after interruption
    ./sweep/run_sweep.py --only max-num-seqs    # only tags containing this
"""
import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN_SH = REPO / "run.sh"
BASE_PARAMS_FILE = REPO / "config" / "vllm-params.yaml"
GENERATED_DIR = REPO / "config" / "generated"
LOGS_DIR = HERE / "logs"
MANIFEST_PATH = HERE / "manifest.jsonl"
NAMESPACE = "vllm-inf"
OWN_GPU_OWNERS = {"vllm-server", "vllm"}

sys.path.insert(0, str(HERE))
from grid import build_plan, sanitize  # noqa: E402


def load_base_manifest():
    with open(BASE_PARAMS_FILE) as f:
        return yaml.safe_load(f)


def generate_configmap(base_manifest, overrides, tag):
    manifest = copy.deepcopy(base_manifest)
    manifest["data"].update(overrides)
    manifest.setdefault("metadata", {}).setdefault("labels", {})[
        "sweep-tag"
    ] = sanitize(tag)[:63]
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / "vllm-params.{}.yaml".format(tag)
    with open(out_path, "w") as f:
        # NB: key order isn't preserved on old PyYAML (no sort_keys kwarg);
        # harmless since a ConfigMap's `data` map has no ordering semantics.
        yaml.safe_dump(manifest, f, default_flow_style=False)
    return out_path


def append_manifest(record):
    with open(MANIFEST_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_manifest_status():
    """Returns {tag: last_status} from prior runs, for --resume."""
    status = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                status[rec["tag"]] = rec["status"]
    return status


def foreign_gpu_holders():
    """Pods in the namespace requesting nvidia.com/gpu that aren't our own
    vllm-server. Returns a list of pod names (empty if the GPU is free/ours)."""
    try:
        out = subprocess.check_output(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "-o", "json"],
            universal_newlines=True,
        )
    except Exception as exc:  # noqa: BLE001
        print("warning: could not check GPU occupancy: {}".format(exc))
        return []
    pods = json.loads(out).get("items", [])
    holders = []
    for pod in pods:
        name = pod.get("metadata", {}).get("name", "")
        app_label = pod.get("metadata", {}).get("labels", {}).get("app", "")
        if app_label in OWN_GPU_OWNERS or name.startswith("vllm-server"):
            continue
        for c in pod.get("spec", {}).get("containers", []):
            if "nvidia.com/gpu" in (c.get("resources", {}).get("requests", {}) or {}):
                phase = pod.get("status", {}).get("phase", "?")
                holders.append("{} ({})".format(name, phase))
                break
    return holders


def wait_for_free_gpu(timeout_s):
    """Poll until no foreign pod is holding the GPU, or timeout_s elapses."""
    deadline = time.time() + timeout_s
    warned = False
    while time.time() < deadline:
        holders = foreign_gpu_holders()
        if not holders:
            return True
        if not warned:
            print(
                "GPU currently held by another workload: {} — waiting up to "
                "{:.0f} min for it to free up before deploying "
                "(this is a shared single-GPU cluster)".format(holders, timeout_s / 60)
            )
            warned = True
        time.sleep(20)
    holders = foreign_gpu_holders()
    if holders:
        print(
            "WARNING: GPU still held by {} after waiting; proceeding anyway "
            "(run.sh's free_gpu will try to scale it down, but it may come "
            "back and cause this config to fail).".format(holders)
        )
    return not holders


def run_one(item, base_manifest, bench, gpu_wait_timeout, attempt=0):
    tag = item["tag"]
    cfg_path = generate_configmap(base_manifest, item["overrides"], tag)

    env = os.environ.copy()
    env["PARAMS"] = str(cfg_path)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if attempt == 0 else ".retry{}".format(attempt)
    log_path = LOGS_DIR / "{}{}.log".format(tag, suffix)

    print("\n{}".format("=" * 78))
    print("[{}] param={} value={} overrides={}".format(
        tag, item["param"], item["value"], item["overrides"]))
    print("  config: {}".format(cfg_path))
    print("  log:    {}".format(log_path))
    print("=" * 78)

    wait_for_free_gpu(gpu_wait_timeout)

    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    start = time.time()
    status = "failed"
    interrupted = False
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                [str(RUN_SH), "all", bench, tag],
                cwd=str(REPO),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    logf.write(line)
                proc.wait()
            except KeyboardInterrupt:
                interrupted = True
                proc.terminate()
                raise
        if proc.returncode == 0:
            status = "success"
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:  # noqa: BLE001
        print("[{}] ERROR launching run.sh: {}".format(tag, exc))

    elapsed = time.time() - start
    if interrupted:
        status = "interrupted"

    if status == "success":
        try:
            subprocess.run(
                [str(RUN_SH), "pull", tag], cwd=str(REPO), env=env, check=True
            )
        except subprocess.CalledProcessError as exc:
            print("[{}] WARNING: pull failed: {}".format(tag, exc))

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"
    record = dict(
        tag=tag,
        param=item["param"],
        key=item["key"],
        value=item["value"],
        overrides=item["overrides"],
        bench=bench,
        status=status,
        elapsed_s=round(elapsed, 1),
        started_at=started_at,
        finished_at=finished_at,
        log=str(log_path),
    )
    append_manifest(record)
    print("[{}] -> {} ({:.1f}s)".format(tag, status.upper(), elapsed))
    if interrupted:
        raise KeyboardInterrupt
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bench", default="both", choices=["rate", "concurrent", "both"],
        help="which benchmark(s) to run per config (default: both)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument(
        "--resume", action="store_true",
        help="skip tags already marked 'success' in sweep/manifest.jsonl",
    )
    ap.add_argument(
        "--only", default=None,
        help="only run configs whose tag contains this substring",
    )
    ap.add_argument(
        "--stop-on-error", action="store_true",
        help="abort the sweep on the first failed config (default: continue)",
    )
    ap.add_argument(
        "--max-retries", type=int, default=1,
        help="retry a failed config this many times (fresh deploy+bench) "
        "before recording it as a final failure (default: 1)",
    )
    ap.add_argument(
        "--gpu-wait-timeout", type=int, default=180,
        help="seconds to wait for a foreign workload to free the shared GPU "
        "before deploying anyway (default: 180)",
    )
    args = ap.parse_args()

    plan = build_plan()
    if args.only:
        plan = [p for p in plan if args.only in p["tag"]]
        if not plan:
            sys.exit("no configs match --only '{}'".format(args.only))

    prior_status = load_manifest_status() if args.resume else {}
    if args.resume:
        plan = [p for p in plan if prior_status.get(p["tag"]) != "success"]

    print("Planned configs ({}):".format(len(plan)))
    for item in plan:
        print("  - {} ({}={})".format(item["tag"], item["param"], item["value"]))
    est_minutes = len(plan) * (28 if args.bench == "both" else (18 if args.bench == "concurrent" else 8))
    print("\nEstimated wall-clock: ~{} min (~{:.1f} h) at current bench={}".format(
        est_minutes, est_minutes / 60.0, args.bench))

    if args.dry_run:
        return

    base_manifest = load_base_manifest()
    results = {"success": [], "failed": []}
    try:
        for item in plan:
            status = "failed"
            for attempt in range(args.max_retries + 1):
                if attempt > 0:
                    print("[{}] retrying (attempt {}/{})".format(
                        item["tag"], attempt + 1, args.max_retries + 1))
                status = run_one(
                    item, base_manifest, args.bench, args.gpu_wait_timeout,
                    attempt=attempt,
                )
                if status == "success":
                    break
            results[status if status in results else "failed"].append(item["tag"])
            if status != "success" and args.stop_on_error:
                print("\nStopping sweep on first failure (--stop-on-error).")
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved in the manifest; rerun with "
              "--resume to continue.")
        return

    print("\n{}".format("=" * 78))
    print("Sweep done: {} succeeded, {} failed".format(
        len(results["success"]), len(results["failed"])))
    if results["failed"]:
        print("Failed tags: {}".format(", ".join(results["failed"])))
    print("Manifest: {}".format(MANIFEST_PATH))
    print("Next: python3 sweep/aggregate.py")


if __name__ == "__main__":
    main()
