"""The `./bench` command line."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import (
    compare,
    doctor,
    kube as kube_mod,
    migrate,
    model_registry,
    plan as plan_mod,
    render,
    report,
    run_queue,
    runner,
    settings,
    spec,
    store,
    util,
    workload_registry,
)

EPILOG = """\
examples:
  ./bench run baseline                       one config, both workloads
  ./bench run ofat-scheduler --dry-run       show the plan and the time it needs
  ./bench run laguna-grid --resume           carry on after an interruption
  ./bench run ofat-scheduler --only mns      just one axis
  ./bench report ofat-scheduler              plots for every config in it
  ./bench runs                               what has been measured, with tags
  ./bench show <tag>                         one run's config, engine facts, files
  ./bench compare <tag> <tag>                head-to-head deltas
  ./bench status                             what the cluster is doing now
"""


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment", help="experiment name (experiments/<name>.yaml) or a path")
    parser.add_argument(
        "--workloads",
        help="comma-separated workload names, overriding the experiment's own "
        f"(available: {', '.join(workload_registry.available())})",
    )
    parser.add_argument(
        "--only",
        help="run only configs whose tag, axis name or value contains this",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip configs already recorded as successful",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="append a suffix to every tag, so a re-run doesn't overwrite the "
        "originals and both can be plotted side by side",
    )
    parser.add_argument(
        "--tag",
        help="override the tag for a single-config experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the wall-clock estimate, touch nothing",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort on the first failed config instead of carrying on",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="retries for a failed config, fresh deploy each time (default: 0). "
        "Configs that fail because they cannot fit in memory are never retried",
    )
    parser.add_argument(
        "--gpu-wait-timeout",
        type=int,
        default=180,
        help="seconds to wait for other workloads to release enough GPUs for this "
        "config's parallelism before deploying anyway (default: 180)",
    )
    parser.add_argument(
        "--no-preseed",
        action="store_true",
        help="skip the weight-cache probe and pre-seed",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="skip the post-deploy check that the server really serves the model",
    )
    parser.add_argument(
        "--quiet-logs",
        action="store_true",
        help="don't echo benchmark output (it is still written to the run directory)",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="benchmark externally hosted OpenAI-compatible endpoints instead of "
        "deploying the in-cluster vLLM server",
    )
    parser.add_argument(
        "--external-endpoint",
        help="default endpoint URL for --external runs "
        "(for example https://host/v1/chat/completions)",
    )
    parser.add_argument(
        "--external-model",
        help="model registry name for --external ad-hoc runs "
        "(used for tokenizer/model identity in benchmark jobs)",
    )
    parser.add_argument(
        "--external-api-key",
        help="API key for OpenAI-compatible external endpoint benchmark jobs",
    )
    parser.add_argument(
        "--no-prompt-endpoint",
        action="store_true",
        help="in --external mode, do not prompt before each run; reuse --external-endpoint",
    )


def _options(args: argparse.Namespace) -> runner.Options:
    return runner.Options(
        workloads=[w.strip() for w in args.workloads.split(",")] if args.workloads else None,
        only=args.only,
        resume=args.resume,
        suffix=args.suffix,
        tag=args.tag,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        max_retries=max(0, args.max_retries),
        gpu_wait_timeout=args.gpu_wait_timeout,
        preseed=not args.no_preseed,
        smoke=not args.no_smoke,
        echo_logs=not args.quiet_logs,
        external=args.external,
        external_endpoint=args.external_endpoint,
        external_model=args.external_model,
        external_api_key=args.external_api_key,
        prompt_endpoint_each_run=not args.no_prompt_endpoint,
    )


def _run_request(experiment: str, options: runner.Options) -> int:
    external_manifest = runner.external_manifest_path(experiment)
    if external_manifest.exists() and runner.is_external_manifest(external_manifest):
        options.external = True
        return runner.run_external_manifest(str(external_manifest), options)

    if options.external:
        if experiment.endswith(".yaml") or experiment.endswith(".yml") or Path(experiment).is_file():
            return runner.run_external_manifest(experiment, options)
        known = set(spec.available())
        bare = {Path(name).name for name in known}
        explicit_file = experiment.endswith(".yaml") or experiment.endswith(".yml")
        qualified = "/" in experiment
        if explicit_file or qualified or experiment in known or experiment in bare:
            loaded = spec.load(experiment)
            return runner.run_experiment(loaded, options)
        return runner.run_external_named(experiment, options)
    loaded = spec.load(experiment)
    return runner.run_experiment(loaded, options)


def _is_external_request(experiment: str, options: runner.Options) -> bool:
    if options.external:
        return True
    external_manifest = runner.external_manifest_path(experiment)
    return external_manifest.exists() and runner.is_external_manifest(external_manifest)


def _drain_queue() -> int:
    final_code = 0
    while True:
        item = run_queue.pop_next()
        if not item:
            return final_code
        util.rule(f"queued run: {item.experiment}  [{item.id}]")
        options = runner.Options(**item.options)
        code = _run_request(item.experiment, options)
        if code != 0:
            final_code = code


def _start_queue_worker() -> int:
    active = run_queue.active_run()
    if active:
        util.die(
            f"a run is already active (pid {active.pid}, experiment '{active.experiment}'); "
            "queue worker is already running"
        )
    queued = run_queue.list_items()
    if not queued:
        util.step("queue is empty")
        return 0
    if not run_queue.try_acquire_active("queue-worker"):
        util.die("another run became active; try again")
    util.step("starting queue worker")
    try:
        return _drain_queue()
    finally:
        run_queue.clear_active()


def cmd_run(args: argparse.Namespace) -> int:
    options = _options(args)
    is_external = _is_external_request(args.experiment, options)
    if args.queue:
        if is_external:
            util.warn("ignoring --queue for external runs; starting now")
            return _run_request(args.experiment, options)
        item, position = run_queue.enqueue(args.experiment, dataclasses.asdict(options))
        util.ok(f"queued {args.experiment} as #{position} ({item.id})")
        # If nothing is running, become the queue worker and start immediately.
        active = run_queue.active_run()
        if not active and run_queue.try_acquire_active(args.experiment):
            util.step("queue is idle; starting queued runs now")
            try:
                return _drain_queue()
            finally:
                run_queue.clear_active()
        return 0

    if is_external:
        return _run_request(args.experiment, options)

    active = run_queue.active_run()
    if active and active.pid != 0:
        util.die(
            f"a run is already active (pid {active.pid}, experiment '{active.experiment}'). "
            "Use --queue to enqueue this request."
        )
    if not run_queue.try_acquire_active(args.experiment):
        util.die("another run became active; use --queue to enqueue this request")
    try:
        code = _run_request(args.experiment, options)
        queued_code = _drain_queue()
        return queued_code if queued_code else code
    finally:
        run_queue.clear_active()


def cmd_queue(args: argparse.Namespace) -> int:
    if args.action == "start":
        return _start_queue_worker()
    if args.action == "rm":
        if args.position is None:
            util.die("name a queue position to remove, for example: ./bench queue rm 2")
        removed = run_queue.remove_at(args.position)
        util.ok(f"removed queue position {args.position}: {removed.experiment} ({removed.id})")
        return 0

    active = run_queue.active_run()
    queued = run_queue.list_items()
    util.rule("run queue")
    if active:
        print(
            util.table(
                ["pid", "host", "started", "experiment"],
                [[active.pid, active.host or "-", active.started_at, active.experiment or "-"]],
            )
        )
    else:
        util.step("no active run")
    print()
    util.rule(f"queued jobs ({len(queued)})")
    if not queued:
        print("(empty)")
        return 0
    print(
        util.table(
            ["pos", "id", "enqueued", "experiment"],
            [
                [i, item.id, item.enqueued_at.replace("T", " ")[:19], item.experiment]
                for i, item in enumerate(queued, 1)
            ],
        )
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Dump the manifests a run would apply, without applying them."""
    experiment = spec.load(args.experiment)
    runs = plan_mod.expand(experiment, suffix="", workloads=None)
    runs = plan_mod.filter_runs(runs, args.only)
    run = runs[0]
    cluster = settings.Cluster.load()
    _, _, model = experiment.resolved()
    print(f"# --- ConfigMap for {run.tag} ---")
    print(render.params_configmap(cluster, run))
    print(f"# --- Deployment + Service ({model.name}, {run.topology.human}) ---")
    print(render.deployment(cluster, model, run))
    for workload in workload_registry.load_many(run.workloads):
        print(f"# --- benchmark Job: {workload.name} ---")
        print(render.benchmark_job(cluster, run, workload))
    if len(runs) > 1:
        util.log(f"({len(runs)} configs planned; showed the first — use --only to pick another)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    # A suffixed group is a separate sweep — often measured against a different
    # workload — so it is excluded from the default view rather than blended in.
    suffix = None if args.all_suffixes else (args.suffix or "")
    status = None if args.include_failed else {store.SUCCESS}

    if args.all:
        records = store.select(status=status, suffix=suffix, model=args.model)
        title = "all runs"
        out_dir = settings.REPORTS_DIR / "all"
    else:
        if not args.experiment:
            known = ", ".join(store.experiments()) or "(nothing recorded yet)"
            util.die(f"name an experiment, or pass --all. Recorded: {known}")
        records = store.select(
            experiment=args.experiment, model=args.model, suffix=suffix, status=status
        )
        if not records:
            available = store.select(experiment=args.experiment)
            if available:
                suffixes = sorted({r.suffix for r in available if r.suffix})
                hint = (
                    f" It does have runs under suffix(es): {', '.join(suffixes)} — "
                    f"add --suffix <name> or --all-suffixes."
                    if suffixes
                    else " Its runs all failed; add --include-failed to see them."
                )
            else:
                hint = f" Recorded experiments: {', '.join(store.experiments()) or '(none)'}"
            util.die(f"no runs to report for '{args.experiment}'.{hint}")
        title = args.experiment
        out_dir = settings.REPORTS_DIR / util.sanitize(args.experiment)
        if suffix:
            out_dir = out_dir / util.sanitize(suffix)

    if suffix == "":
        hidden = {
            r.suffix
            for r in store.select(
                experiment=None if args.all else args.experiment, status=status
            )
            if r.suffix
        }
        if hidden:
            util.step(
                f"excluding suffixed run group(s) {', '.join(sorted(hidden))} from this "
                f"view (--suffix <name> to plot one, --all-suffixes for everything)"
            )

    report.build(records, out_dir, title, param=args.param)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """List recorded runs — mainly so you can find the tags `compare` wants."""
    status = None if args.include_failed else {store.SUCCESS}
    records = store.select(experiment=args.experiment, model=args.model, status=status)
    if args.match:
        records = [r for r in records if args.match in r.tag]
    if not records:
        util.die(
            "no runs match. Recorded experiments: "
            f"{', '.join(store.experiments()) or '(none)'}"
        )
    records.sort(key=lambda r: r.finished_at or r.started_at, reverse=True)
    rows = [
        [
            r.tag,
            r.experiment,
            r.model,
            r.topology.label,
            r.status,
            ", ".join(sorted(r.results or {})) or "-",
            (r.finished_at or "")[:16].replace("T", " "),
        ]
        for r in records
    ]
    util.rule(f"{len(rows)} run(s)")
    print(
        util.table(
            ["tag", "experiment", "model", "gpus", "status", "workloads", "finished"], rows
        )
    )
    util.step("head-to-head: ./bench compare <tag> <tag>  (a unique substring works too)")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Everything recorded about one run: config as applied, engine, artefacts.

    The index holds all of this per attempt, but reading JSONL by hand to answer
    "what was this run actually given, and what did the engine make of it" is not
    a reasonable ask.
    """
    record = store.resolve_tag(args.tag)
    run_dir = settings.REPO / record.run_dir if record.run_dir else store.run_dir(record.tag)

    util.rule(record.tag)
    facts = [
        ["experiment", f"{record.experiment} ({record.strategy})"],
        ["model", f"{record.model}  [{record.hf_repo or '?'}]"],
        ["gpus", record.topology.label],
        ["axis", f"{record.param} = {record.value}" if record.param else "-"],
        ["suffix", record.suffix or "-"],
        ["status", record.status + (f" ({record.failure_kind})" if record.failure_kind else "")],
        ["attempt", record.attempt or "-"],
        ["started", (record.started_at or "-")[:19].replace("T", " ")],
        ["took", util.human_duration(record.elapsed_s)],
        ["workloads", ", ".join(record.workloads) or "-"],
        ["results", ", ".join(sorted(record.results or {})) or "none written"],
    ]
    print(util.table(["", ""], facts))

    if record.failure_detail:
        print()
        util.warn(record.failure_detail)

    print()
    util.rule("engine, read from the server log at startup")
    if record.engine:
        print(util.table(["fact", "value"], sorted(record.engine.items())))
    else:
        util.step(
            "nothing recorded — either the run predates this tool (see `bench migrate`) "
            "or the server never came up"
        )
    if record.smoke:
        util.step(f"smoke test: {record.smoke}")

    print()
    util.rule("config as applied (* = set by this run, not the base)")
    extra_args = record.params.get("EXTRA_ARGS", record.extra_args)
    rows = [
        ["*" if key in record.overrides else "", key, util.elide(str(value), 60)]
        for key, value in sorted(record.params.items())
        if key != "EXTRA_ARGS"
    ]
    print(util.table(["", "key", "value"], rows) if rows else "  (no config recorded)")
    if extra_args:
        print()
        print("  EXTRA_ARGS:")
        for flag in extra_args.split():
            print(f"    {flag}")

    print()
    util.rule("files")
    if run_dir.exists():
        files = sorted(run_dir.iterdir(), key=lambda p: p.name)
        print(
            util.table(
                ["file", "size", "what"],
                [[p.name, f"{p.stat().st_size / 1024:.0f}K", _blurb(p)] for p in files],
            )
        )
        util.step(f"in {run_dir}")
    else:
        util.step(f"{run_dir} does not exist")
    return 0


# The per-run directory is only self-explanatory if you built it.
_FILE_BLURB = {
    "summary.json": "parsed metrics, cached for `bench report`",
    "config.yaml": "the exact ConfigMap this run was served with",
    "server.log": "vLLM startup log — where the engine facts above come from",
    "engine.log": "vLLM log captured while the benchmark was running",
}
_SUFFIX_BLURB = {
    ".json": "raw GuideLLM results",
    ".html": "GuideLLM's own report — open it in a browser",
    ".log": "benchmark job output",
}


def _blurb(path: Path) -> str:
    return _FILE_BLURB.get(path.name) or _SUFFIX_BLURB.get(path.suffix, "")


def cmd_compare(args: argparse.Namespace) -> int:
    out_dir = settings.REPORTS_DIR / "compare"
    compare.run(
        args.tag_a,
        args.tag_b,
        out_dir,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    kube = kube_mod.Kube(settings.Cluster.load())
    util.rule("cluster")
    for line in kube.status_lines():
        print(line)
    jobs = kube.bench_jobs()
    print()
    util.rule("benchmark jobs")
    print(util.table(["job", "state", "created"], jobs))

    records = store.latest_by_tag()
    if records:
        recent = sorted(records.values(), key=lambda r: r.finished_at or "", reverse=True)[:8]
        print()
        util.rule("most recent runs")
        print(
            util.table(
                ["tag", "experiment", "model", "status", "took"],
                [
                    [r.tag, r.experiment, r.model, r.status, util.human_duration(r.elapsed_s)]
                    for r in recent
                ],
            )
        )
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    kube = kube_mod.Kube(settings.Cluster.load())
    glob = f"*{args.tag}*" if args.tag else "*"
    destination = Path(args.dest) if args.dest else settings.RUNS_DIR / "_pulled"
    pulled = kube.pull(glob, destination)
    util.log(f"pulled {len(pulled)} file(s) -> {destination}")
    if pulled:
        util.step(
            "these are raw PVC files; a normal `bench run` files them into "
            "runs/<tag>/ automatically"
        )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    if args.preseed:
        model = model_registry.load(args.preseed)
        kube = kube_mod.Kube(settings.Cluster.load())
        kube.ensure_namespace()
        kube.ensure_secret()
        kube.ensure_model_cache_pvc(model)
        cached = kube.model_is_cached(model)
        if cached and not args.force:
            util.ok(f"{model.name} is already cached on {model.cache_pvc}")
            return 0
        return 0 if kube.preseed(model) else 1

    rows = []
    for name in model_registry.available():
        model = model_registry.load(name)
        rows.append(
            [
                model.name,
                model.hf_repo,
                model.cache_pvc,
                model.params.get("MAX_MODEL_LEN", "-"),
                model.image or "(cluster default)",
            ]
        )
    util.rule("models")
    print(util.table(["name", "hf repo", "cache pvc", "max model len", "image"], rows))
    print()
    util.rule("workloads")
    print(
        util.table(
            ["name", "shape", "load levels", "≈ time"],
            [
                [
                    w.name,
                    w.mode,
                    ", ".join(str(int(t)) for t in w.targets),
                    util.human_duration(w.est_seconds),
                ]
                for w in (workload_registry.load(n) for n in workload_registry.available())
            ],
        )
    )
    print()
    util.rule("experiments")
    rows = []
    for name in spec.available():
        experiment = spec.load(name)
        runs = plan_mod.expand(experiment)
        rows.append(
            [
                experiment.name,
                spec.group(experiment.path),
                experiment.kind,
                experiment.model,
                plan_mod.topology_summary(runs),
                len(runs),
                util.human_duration(plan_mod.estimate_seconds(runs)),
            ]
        )
    print(
        util.table(
            ["name", "group", "kind", "model", "gpus", "configs", "≈ time"], rows
        )
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    return doctor.run(fix=args.fix)


def cmd_migrate(args: argparse.Namespace) -> int:
    return migrate.run(dry_run=args.dry_run)


def cmd_clean(args: argparse.Namespace) -> int:
    kube = kube_mod.Kube(settings.Cluster.load())
    if args.what == "jobs":
        print(kube.clean_jobs())
        return 0
    glob = f"*{args.tag}*" if args.tag else "*"
    util.warn(
        f"about to delete '{glob}' from PVC '{kube.cluster.results_pvc}'. "
        f"This is permanent — pull anything you still want first."
    )
    if input("Type 'yes' to continue: ").strip() != "yes":
        util.die("aborted")
    print(kube.clean_results(glob))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./bench",
        description="Benchmark vLLM serving configs on the cluster: single runs, "
        "one-factor-at-a-time sweeps and grid searches, from one place.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="deploy and benchmark every config in an experiment",
        description="Deploy each config in turn, run its workloads, pull the "
        "results and record them. Safe to interrupt: rerun with --resume.",
    )
    _add_run_args(run_parser)
    run_parser.set_defaults(func=cmd_run)
    run_parser.add_argument(
        "--queue",
        action="store_true",
        help="enqueue this run to execute after the currently active run/sweep",
    )

    render_parser = subparsers.add_parser(
        "render", help="print the manifests a run would apply, without applying them"
    )
    render_parser.add_argument("experiment")
    render_parser.add_argument("--only", help="pick a specific config from the plan")
    render_parser.set_defaults(func=cmd_render)

    report_parser = subparsers.add_parser(
        "report", help="plots for recorded runs"
    )
    report_parser.add_argument("experiment", nargs="?", help="experiment to report on")
    report_parser.add_argument("--all", action="store_true", help="every recorded run")
    report_parser.add_argument("--param", help="drill into one swept axis")
    report_parser.add_argument("--model", help="restrict to one model")
    report_parser.add_argument(
        "--suffix",
        help="plot the runs carrying this tag suffix instead of the base runs",
    )
    report_parser.add_argument(
        "--all-suffixes",
        action="store_true",
        help="plot base and suffixed runs together (they may not be comparable)",
    )
    report_parser.add_argument(
        "--include-failed",
        action="store_true",
        help="also plot runs that failed (partial results)",
    )
    report_parser.set_defaults(func=cmd_report)

    runs_parser = subparsers.add_parser("runs", help="list recorded runs and their tags")
    runs_parser.add_argument("match", nargs="?", help="only tags containing this")
    runs_parser.add_argument("--experiment", help="restrict to one experiment")
    runs_parser.add_argument("--model", help="restrict to one model")
    runs_parser.add_argument(
        "--include-failed", action="store_true", help="also list runs that did not succeed"
    )
    runs_parser.set_defaults(func=cmd_runs)

    show_parser = subparsers.add_parser(
        "show", help="everything recorded about one run: config, engine facts, files"
    )
    show_parser.add_argument("tag", help="run tag, or a unique substring of one")
    show_parser.set_defaults(func=cmd_show)

    compare_parser = subparsers.add_parser(
        "compare", help="head-to-head deltas between two runs"
    )
    compare_parser.add_argument("tag_a", help="baseline run (tag or unique substring)")
    compare_parser.add_argument("tag_b", help="candidate run (tag or unique substring)")
    compare_parser.add_argument("--label-a", help="legend label for the first run")
    compare_parser.add_argument("--label-b", help="legend label for the second run")
    compare_parser.set_defaults(func=cmd_compare)

    status_parser = subparsers.add_parser("status", help="server, jobs and recent runs")
    status_parser.set_defaults(func=cmd_status)

    queue_parser = subparsers.add_parser(
        "queue", help="show queue, start worker, or remove queued jobs by position"
    )
    queue_parser.add_argument(
        "action",
        nargs="?",
        choices=["start", "rm"],
        help="'start' runs queued jobs now when idle; 'rm' removes one queued position",
    )
    queue_parser.add_argument(
        "position",
        nargs="?",
        type=int,
        help="queue position to remove (used with: ./bench queue rm <position>)",
    )
    queue_parser.set_defaults(func=cmd_queue)

    pull_parser = subparsers.add_parser(
        "pull", help="copy raw result files off the results PVC"
    )
    pull_parser.add_argument("tag", nargs="?", help="only files matching this substring")
    pull_parser.add_argument("--dest", help="destination directory")
    pull_parser.set_defaults(func=cmd_pull)

    models_parser = subparsers.add_parser(
        "models", help="list models, workloads and experiments; pre-seed weights"
    )
    models_parser.add_argument("--preseed", metavar="MODEL", help="download a model's weights")
    models_parser.add_argument(
        "--force", action="store_true", help="pre-seed even if the weights look cached"
    )
    models_parser.set_defaults(func=cmd_models)

    doctor_parser = subparsers.add_parser(
        "doctor", help="audit the index against the files on disk"
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="mark runs whose results are unusable as failed, so --resume redoes them",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    migrate_parser = subparsers.add_parser(
        "migrate", help="import pre-refactor results into the run store (one-off)"
    )
    migrate_parser.add_argument("--dry-run", action="store_true")
    migrate_parser.set_defaults(func=cmd_migrate)

    clean_parser = subparsers.add_parser("clean", help="delete benchmark jobs or PVC results")
    clean_parser.add_argument(
        "what", choices=["jobs", "results"], help="'jobs' deletes Job objects; 'results' wipes the PVC"
    )
    clean_parser.add_argument("tag", nargs="?", help="restrict 'results' to matching files")
    clean_parser.set_defaults(func=cmd_clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except kube_mod.KubectlError as exc:
        util.die(str(exc))
    except KeyboardInterrupt:
        util.warn("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())