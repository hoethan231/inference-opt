#!/usr/bin/env python3
"""
check_results.py - audits sweep/manifest.jsonl against the actual pulled
result files in results/.

Why this exists: run.sh's run_one() used to only print a warning when a
benchmark Job failed on the cluster -- it never returned a non-zero exit
code. That's now fixed, but every manifest entry recorded *before* the fix
may say status: "success" even though the benchmark Job actually failed
(most commonly because the old shared results PVC was full and GuideLLM
crashed trying to save its report). This script finds those mismatches by
checking whether the actual results/vllm-{rate,users}-<tag>.json files
exist, are non-empty, and parse as JSON.

Usage:
    python3 sweep/check_results.py                # report only
    python3 sweep/check_results.py --fix-manifest  # also append corrected
                                                    # "failed" entries so
                                                    # `run_sweep.py --resume`
                                                    # will pick them back up
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_DIR = REPO / "results/outputs"
MANIFEST_PATH = HERE / "manifest.jsonl"

sys.path.insert(0, str(HERE))
from grid import build_plan  # noqa: E402


def file_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False
    return True


def load_last_manifest():
    """Returns {tag: last_record} -- last line per tag wins, matching how
    run_sweep.py's own --resume logic reads this append-only file."""
    records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[rec["tag"]] = rec
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fix-manifest", action="store_true",
        help="append corrected 'failed' entries for tags marked 'success' "
        "whose result file(s) are actually missing/empty/invalid",
    )
    args = ap.parse_args()

    plan = build_plan()
    last = load_last_manifest()

    good, bad, never_run = [], [], []
    for item in plan:
        tag = item["tag"]
        rec = last.get(tag)
        bench = rec["bench"] if rec else "both"
        needed = []
        if bench in ("rate", "both"):
            needed.append(RESULTS_DIR / "vllm-rate-{}.json".format(tag))
        if bench in ("concurrent", "both"):
            needed.append(RESULTS_DIR / "vllm-users-{}.json".format(tag))
        missing = [p.name for p in needed if not file_ok(p)]
        if rec is None:
            never_run.append(tag)
        elif missing:
            bad.append((tag, rec["status"], missing))
        else:
            good.append(tag)

    print("{:<40} {:<14} issue".format("tag", "manifest status"))
    print("-" * 90)
    for tag in never_run:
        print("{:<40} {:<14}".format(tag, "(never run)"))
    for tag, status, missing in bad:
        print("{:<40} {:<14} missing/empty: {}".format(tag, status, ", ".join(missing)))
    print("-" * 90)
    print("{} clean, {} need rerun, {} never attempted (of {} planned)".format(
        len(good), len(bad), len(never_run), len(plan)))

    if args.fix_manifest:
        to_fix = [tag for tag, status, _ in bad if status == "success"]
        if not to_fix:
            print("\nNothing to fix -- no 'success'-marked tag has bad result files.")
            return
        now = datetime.datetime.utcnow().isoformat() + "Z"
        by_tag = {p["tag"]: p for p in plan}
        with open(MANIFEST_PATH, "a") as f:
            for tag in to_fix:
                item = by_tag[tag]
                rec = dict(
                    tag=tag,
                    param=item["param"],
                    key=item["key"],
                    value=item["value"],
                    overrides=item["overrides"],
                    bench=last[tag]["bench"],
                    status="failed",
                    elapsed_s=0.0,
                    started_at=now,
                    finished_at=now,
                    log=None,
                    note="corrected by check_results.py --fix-manifest: result "
                    "file(s) missing/empty/invalid despite an earlier 'success' "
                    "status (predates the run.sh exit-code fix)",
                )
                f.write(json.dumps(rec) + "\n")
        print("\nAppended {} corrected 'failed' entries to {}.".format(
            len(to_fix), MANIFEST_PATH))
        print("Now run: python3 sweep/run_sweep.py --resume")


if __name__ == "__main__":
    main()
