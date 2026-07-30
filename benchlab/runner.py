"""Execute a plan: deploy, benchmark, pull, record. Resumable and retrying."""

from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path

from . import ingest
from . import kube as kube_mod
from . import plan as plan_mod
from . import render, settings, store, util, workload_registry
from .model_registry import Model
from .plan import Run
from .spec import Experiment
from .store import RunRecord
from .workload_registry import Workload

# Deploy failures that mean "this config cannot run here", not "try again".
INFEASIBLE_REASONS = {"oom"}


@dataclasses.dataclass
class Options:
    workloads: list[str] | None = None
    only: str | None = None
    resume: bool = False
    suffix: str = ""
    tag: str | None = None
    dry_run: bool = False
    stop_on_error: bool = False
    max_retries: int = 1
    gpu_wait_timeout: int = 180
    preseed: bool = True
    smoke: bool = True
    echo_logs: bool = True


def run_experiment(experiment: Experiment, options: Options) -> int:
    runs = plan_mod.expand(
        experiment,
        suffix=options.suffix,
        workloads=options.workloads,
        tag_override=options.tag,
    )
    runs = plan_mod.filter_runs(runs, options.only)

    if options.resume:
        done = store.successful_tags()
        skipped = [r.tag for r in runs if r.tag in done]
        runs = [r for r in runs if r.tag not in done]
        if skipped:
            util.log(f"--resume: skipping {len(skipped)} already-successful run(s)")
        if not runs:
            util.ok("nothing left to run; every planned config already succeeded")
            return 0

    print(plan_mod.describe(runs, experiment))
    print()
    if options.dry_run:
        util.log("dry run: nothing was applied to the cluster")
        return 0

    workloads = workload_registry.load_many(runs[0].workloads)
    _, _, model = experiment.resolved()
    kube = kube_mod.Kube(settings.Cluster.load())

    _preflight(kube, model, workloads, options)

    outcomes: dict[str, list[str]] = {"success": [], "failed": [], "infeasible": []}
    try:
        for index, run in enumerate(runs, 1):
            util.rule(f"[{index}/{len(runs)}] {run.tag}   {run.summary()}")
            status = _run_with_retries(kube, run, model, workloads, options)
            outcomes.setdefault(status, []).append(run.tag)
            if status != store.SUCCESS and options.stop_on_error:
                util.warn("stopping on first failure (--stop-on-error)")
                break
    except KeyboardInterrupt:
        util.warn("interrupted; progress is recorded — rerun with --resume to continue")
        return 130

    print()
    util.rule("Summary")
    print(
        f"{len(outcomes['success'])} succeeded, {len(outcomes['failed'])} failed, "
        f"{len(outcomes['infeasible'])} infeasible"
    )
    for label in ("failed", "infeasible"):
        if outcomes[label]:
            print(f"  {label}: {', '.join(outcomes[label])}")
    print()
    util.log(f"next: ./bench report {experiment.name}")
    return 0 if not outcomes["failed"] else 1


def _preflight(
    kube: kube_mod.Kube, model: Model, workloads: list[Workload], options: Options
) -> None:
    util.log("preflight")
    kube.ensure_secret()
    kube.ensure_results_pvc()
    kube.apply_workloads(workloads)
    util.step(f"workloads loaded: {', '.join(w.name for w in workloads)}")
    if not options.preseed:
        return
    cached = kube.model_is_cached(model)
    if cached is False:
        util.step(f"{model.name} is not on {model.cache_pvc} yet")
        kube.preseed(model)
    elif cached:
        util.step(f"weights for {model.name} are already cached")


def _run_with_retries(
    kube: kube_mod.Kube,
    run: Run,
    model: Model,
    workloads: list[Workload],
    options: Options,
) -> str:
    status = store.FAILED
    for attempt in range(options.max_retries + 1):
        if attempt:
            util.warn(f"retry {attempt}/{options.max_retries} for {run.tag}")
        status = _execute(kube, run, model, workloads, options, attempt)
        if status in (store.SUCCESS, store.INFEASIBLE):
            break  # a config that cannot fit will not fit on a second try
    return status


def _execute(
    kube: kube_mod.Kube,
    run: Run,
    model: Model,
    workloads: list[Workload],
    options: Options,
    attempt: int,
) -> str:
    run_dir = store.run_dir(run.tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    record = RunRecord(
        tag=run.tag,
        experiment=run.experiment,
        strategy=run.strategy,
        model=run.model,
        hf_repo=run.hf_repo,
        param=run.param,
        value=run.value,
        label=run.label,
        suffix=run.suffix,
        overrides=run.overrides,
        params=run.params,
        extra_args=run.extra_args,
        extra_args_append=run.extra_args_append,
        workloads=[w.name for w in workloads],
        attempt=attempt,
        started_at=store.utc_now(),
        run_dir=str(run_dir.relative_to(settings.REPO)),
    )
    started = time.time()

    def finish(status: str, *, kind: str = "", detail: str = "") -> str:
        record.status = status
        record.failure_kind = kind
        record.failure_detail = detail[:500]
        record.elapsed_s = round(time.time() - started, 1)
        record.finished_at = store.utc_now()
        store.append(record)
        marker = util.ok if status == store.SUCCESS else util.warn
        marker(f"{run.tag} -> {status.upper()} ({util.human_duration(record.elapsed_s)})")
        return status

    try:
        # 1. config -> ConfigMap, saved next to the results it produced.
        manifest = render.params_configmap(settings.Cluster.load(), run)
        (run_dir / "config.yaml").write_text(manifest)

        kube.wait_for_free_gpu(options.gpu_wait_timeout)
        kube.free_gpu()
        util.step("applying config and restarting the server")
        kube.apply_params(run)

        # 2. deploy, aborting early on a config that cannot start.
        outcome = kube.deploy(
            run, model, timeout_s=settings.Cluster.load().timeout("rollout_s", 3600)
        )
        if outcome.log:
            (run_dir / "server.log").write_text(outcome.log)
        record.engine = outcome.facts
        if not outcome.ok:
            status = (
                store.INFEASIBLE if outcome.reason in INFEASIBLE_REASONS else store.FAILED
            )
            util.warn(f"deploy failed [{outcome.reason}]: {outcome.detail}")
            return finish(status, kind=outcome.reason, detail=outcome.detail)
        if record.engine:
            util.step(
                "engine: "
                + ", ".join(f"{k}={v}" for k, v in sorted(record.engine.items()))
            )

        # 3. prove it serves the model before spending an hour measuring it.
        if options.smoke:
            passed, detail = kube.smoke_test(model)
            record.smoke = detail[:300]
            if not passed:
                util.warn(f"smoke test failed: {detail}")
                return finish(store.FAILED, kind="smoke", detail=detail)
            util.step(f"smoke test passed ({detail})")

        # 4. benchmark every workload against this one deploy.
        failures = []
        for workload in workloads:
            job = kube.run_benchmark(
                run, workload, run_dir / f"{workload.name}.log", echo=options.echo_logs
            )
            if not job.ok:
                failures.append(f"{workload.name}: {job.detail}")
                util.warn(f"workload '{workload.name}' failed: {job.detail}")

        # 5. pull whatever landed, even on failure — partial data is still data.
        record.results = _collect_results(kube, run, workloads, run_dir)
        missing = [w.name for w in workloads if w.name not in record.results]
        if failures or missing:
            detail = "; ".join(failures) or f"no result file for {', '.join(missing)}"
            return finish(store.FAILED, kind="bench", detail=detail)
        return finish(store.SUCCESS)

    except KeyboardInterrupt:
        finish(store.INTERRUPTED, kind="interrupted", detail="cancelled by user")
        raise
    except kube_mod.KubectlError as exc:
        return finish(store.FAILED, kind="kubectl", detail=str(exc))


def _collect_results(
    kube: kube_mod.Kube, run: Run, workloads: list[Workload], run_dir: Path
) -> dict[str, str]:
    """Pull this run's files off the PVC and file them under the run directory."""
    util.step("pulling results")
    kube.pull(f"{run.tag}__*", run_dir, quiet=False)
    results: dict[str, str] = {}
    for workload in workloads:
        stem = render.output_base(run, workload)
        for extension in ("json", "html"):
            source = run_dir / f"{stem}.{extension}"
            if source.exists():
                shutil.move(str(source), str(run_dir / f"{workload.name}.{extension}"))
        candidate = run_dir / f"{workload.name}.json"
        if ingest.result_is_usable(candidate):
            results[workload.name] = str(candidate.relative_to(settings.REPO))
    return results
