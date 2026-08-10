"""Execute a plan: deploy, benchmark, pull, record. Resumable and retrying."""

from __future__ import annotations

import dataclasses
import shutil
import time
from urllib.parse import urlparse
from pathlib import Path

import yaml

from . import ingest
from . import kube as kube_mod
from . import plan as plan_mod
from . import render, settings, store, util, workload_registry
from . import model_registry
from .model_registry import Model
from .parallelism import Topology
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
    max_retries: int = 0
    gpu_wait_timeout: int = 180
    preseed: bool = True
    smoke: bool = True
    echo_logs: bool = True
    external: bool = False
    external_endpoint: str | None = None
    external_model: str | None = None
    external_api_key: str | None = None
    prompt_endpoint_each_run: bool = True


EXTERNAL_EXPERIMENT = "external"


@dataclasses.dataclass
class ExternalManifest:
    name: str
    model: str
    endpoint: str
    api_key: str = ""
    request_model: str = ""
    workloads: list[str] = dataclasses.field(default_factory=lambda: ["rate", "concurrent"])
    runs: list[str] = dataclasses.field(default_factory=list)


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


def run_external_named(name: str, options: Options) -> int:
    if not options.external:
        util.die("internal error: run_external_named called without --external")
    run = _build_external_run(name, options)
    if options.resume and run.tag in store.successful_tags():
        util.ok(f"--resume: '{run.tag}' already succeeded")
        return 0
    util.rule(f"external run: {run.tag}")
    print(
        util.table(
            ["field", "value"],
            [
                ["experiment", run.experiment],
                ["workloads", ", ".join(run.workloads)],
                ["tag", run.tag],
                ["axes", ", ".join(f"{k}={v}" for k, v in sorted(run.overrides.items())) or "-"],
                ["gpus", Topology().human],
            ],
        )
    )
    print()
    if options.dry_run:
        util.log("dry run: nothing was applied to the cluster")
        return 0

    workloads = workload_registry.load_many(run.workloads)
    kube = kube_mod.Kube(settings.Cluster.load())
    model = _resolve_external_model(options)
    _preflight(kube, model, workloads, options)
    status = _run_with_retries(kube, run, model, workloads, options)
    print()
    util.rule("Summary")
    print(f"{run.tag}: {status}")
    print()
    util.log(f"next: ./bench report {EXTERNAL_EXPERIMENT}")
    return 0 if status == store.SUCCESS else 1


def run_external_manifest(path_or_name: str, options: Options) -> int:
    if not options.external:
        util.die("internal error: run_external_manifest called without --external")
    manifest = _load_external_manifest(path_or_name)
    workloads = workload_registry.load_many(manifest.workloads)
    model = model_registry.load(manifest.model)
    kube = kube_mod.Kube(settings.Cluster.load())
    _preflight(kube, model, workloads, options)

    run_names = manifest.runs or [manifest.name]
    runs = [
        _build_external_run(
            name,
            options,
            experiment=manifest.name,
            model=model,
            request_model=manifest.request_model or "",
        )
        for name in run_names
    ]

    util.rule(f"external manifest: {manifest.name}")
    print(
        util.table(
            ["field", "value"],
            [
                ["model", manifest.model],
                ["endpoint", manifest.endpoint],
                ["workloads", ", ".join(manifest.workloads)],
                ["runs", str(len(runs))],
            ],
        )
    )
    print()
    if options.dry_run:
        util.log("dry run: nothing was applied to the cluster")
        return 0

    outcomes: dict[str, list[str]] = {"success": [], "failed": [], "infeasible": []}
    previous_endpoint = options.external_endpoint
    previous_api = options.external_api_key
    previous_prompt = options.prompt_endpoint_each_run
    try:
        options.external_endpoint = manifest.endpoint
        options.external_api_key = options.external_api_key or manifest.api_key
        options.prompt_endpoint_each_run = False
        for index, run in enumerate(runs, 1):
            util.rule(f"[{index}/{len(runs)}] {run.tag}   {run.summary()}")
            status = _run_with_retries(kube, run, model, workloads, options)
            outcomes.setdefault(status, []).append(run.tag)
            if status != store.SUCCESS and options.stop_on_error:
                util.warn("stopping on first failure (--stop-on-error)")
                break
    finally:
        options.external_endpoint = previous_endpoint
        options.external_api_key = previous_api
        options.prompt_endpoint_each_run = previous_prompt

    print()
    util.rule("Summary")
    print(
        f"{len(outcomes['success'])} succeeded, {len(outcomes['failed'])} failed, "
        f"{len(outcomes['infeasible'])} infeasible"
    )
    print()
    util.log(f"next: ./bench report {manifest.name}")
    return 0 if not outcomes["failed"] else 1


def _preflight(
    kube: kube_mod.Kube, model: Model, workloads: list[Workload], options: Options
) -> None:
    util.log("preflight")
    kube.ensure_namespace()
    kube.ensure_secret()
    kube.ensure_model_cache_pvc(model)
    kube.ensure_results_pvc()
    # Workloads live in one shared ConfigMap; when runs happen in parallel,
    # writing only this run's subset can remove another run's scenario file.
    all_workloads = workload_registry.load_many(workload_registry.available())
    kube.apply_workloads(all_workloads)
    util.step(f"workloads loaded: {', '.join(w.name for w in workloads)}")
    if options.external:
        util.step("external mode: skipping model cache probe, deploy and smoke test")
        return
    if not options.preseed:
        return
    # Pre-seed jobs run on the GPU node with substantial CPU/RAM requests.
    # On smaller clusters an old crashlooping server can block scheduling, so
    # free engine resources before cache probing/pre-seed.
    kube.free_gpu()
    kube.teardown_server()
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
    run_dir = _run_dir(run, options)
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
    endpoint = ""

    def finish(status: str, *, kind: str = "", detail: str = "") -> str:
        record.status = status
        record.failure_kind = kind
        record.failure_detail = detail[:500]
        record.elapsed_s = round(time.time() - started, 1)
        record.finished_at = store.utc_now()
        if endpoint:
            record.note = f"endpoint={endpoint}"
        store.append(record)
        marker = util.ok if status == store.SUCCESS else util.warn
        marker(f"{run.tag} -> {status.upper()} ({util.human_duration(record.elapsed_s)})")
        return status

    try:
        # 1. config -> ConfigMap, saved next to the results it produced.
        manifest = render.params_configmap(settings.Cluster.load(), run)
        (run_dir / "config.yaml").write_text(manifest)

        if options.external:
            endpoint = _resolve_external_endpoint(run.tag, options)
            util.step(f"external endpoint for this run: {endpoint}")
            util.step("external mode: leaving vllm-params untouched")
        else:
            kube.ensure_gpus(run.topology.gpus, options.gpu_wait_timeout)
            util.step(f"applying config and restarting the server on {run.topology.human}")
            kube.apply_params(run)

        if not options.external:
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
        if options.smoke and not options.external:
            passed, detail = kube.smoke_test(model)
            record.smoke = detail[:300]
            if not passed:
                util.warn(f"smoke test failed: {detail}")
                return finish(store.FAILED, kind="smoke", detail=detail)
            util.step(f"smoke test passed ({detail})")

        # 4. benchmark every workload against this one deploy, recording what the
        #    engine does while the load is on it.
        failures = []
        trace = kube.engine_trace(run_dir / "engine.log")
        if options.external:
            trace = _noop_trace()
        with trace:
            for workload in workloads:
                job = kube.run_benchmark(
                    run,
                    workload,
                    run_dir / f"{workload.name}.log",
                    echo=options.echo_logs,
                    service_url=endpoint or None,
                    api_key=options.external_api_key if options.external else None,
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
    cluster = settings.Cluster.load()
    pvc = cluster.external_results_pvc if run.strategy == "external" else cluster.results_pvc
    kube.pull(f"{run.tag}__*", run_dir, quiet=False, pvc=pvc)
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


def _run_dir(run: Run, options: Options) -> Path:
    return store.run_dir(run.tag)


def _resolve_external_model(options: Options) -> Model:
    if options.external_model:
        return model_registry.load(options.external_model)
    known = model_registry.available()
    if len(known) == 1:
        return model_registry.load(known[0])
    util.die(
        "external ad-hoc runs need --external-model <name> "
        f"(available: {', '.join(known) or '(none)'})"
    )


def _build_external_run(
    name: str,
    options: Options,
    *,
    experiment: str = EXTERNAL_EXPERIMENT,
    model: Model | None = None,
    request_model: str = "",
) -> Run:
    if not name:
        util.die("name the run, for example: mns-64-mnbt-1024")
    sanitized = util.sanitize(name)
    suffix = util.sanitize(options.suffix) if options.suffix else ""
    prefixed = sanitized if sanitized.startswith("external-") else f"external-{sanitized}"
    base_tag = prefixed
    tag = f"{prefixed}-{suffix}" if suffix else prefixed
    tag = util.fit_name(tag, settings.MAX_TAG_LEN)

    axes = _parse_external_axes(sanitized)
    overrides = {k: v for k, v in axes}
    label = ", ".join(f"{k}={v}" for k, v in axes) if axes else sanitized

    workloads = options.workloads or ["rate", "concurrent"]
    workload_registry.load_many(workloads)

    model = model or _resolve_external_model(options)
    base_raw = settings.PARAMS_BASE_YAML.read_text()
    parsed = yaml.safe_load(base_raw) or {}
    params = {k: str(v) for k, v in (parsed.get("params") or {}).items()}
    params.update(model.params)
    params.update(model.identity_params())
    if request_model:
        # External gateways often expose model IDs unrelated to our local registry.
        # Only override SERVED_MODEL_NAME (the name sent in API requests).
        # MODEL_NAME must stay as the HF repo so the tokenizer can load it.
        params["SERVED_MODEL_NAME"] = request_model
    extra_args = [str(a) for a in (parsed.get("extra_args") or [])] + model.extra_args
    params["EXTRA_ARGS"] = " ".join(a for a in extra_args if a).strip()

    return Run(
        tag=tag,
        base_tag=base_tag,
        suffix=options.suffix,
        experiment=experiment,
        strategy="external",
        model=model.name,
        hf_repo=model.hf_repo,
        param="external",
        value=sanitized,
        label=label,
        overrides=overrides,
        extra_args_append="",
        params=params,
        extra_args=params.get("EXTRA_ARGS", ""),
        workloads=list(workloads),
    )


def _parse_external_axes(tag: str) -> list[tuple[str, str]]:
    parts = [p for p in tag.split("-") if p]
    if len(parts) < 2:
        return []
    if len(parts) % 2 != 0:
        util.warn(
            f"external tag '{tag}' is not param-value pairs; expected "
            "pattern like mns-64-mnbt-1024"
        )
        return []
    axes = []
    for i in range(0, len(parts), 2):
        axes.append((parts[i], parts[i + 1]))
    return axes


@dataclasses.dataclass
class _ExternalEndpointState:
    last: str = ""


_EXTERNAL_ENDPOINT = _ExternalEndpointState()


def _resolve_external_endpoint(tag: str, options: Options) -> str:
    default = options.external_endpoint or _EXTERNAL_ENDPOINT.last
    if options.prompt_endpoint_each_run:
        prompt = f"External endpoint for '{tag}'"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        entered = input(prompt).strip()
        endpoint = entered or default
    else:
        endpoint = default
    if not endpoint:
        util.die(
            "external mode needs an endpoint. Provide --external-endpoint and/or "
            "answer the prompt."
        )
    _validate_endpoint(endpoint)
    _EXTERNAL_ENDPOINT.last = endpoint
    return endpoint


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        util.die(
            f"invalid endpoint '{endpoint}'. Expected a full URL like "
            "https://host or https://host/prefix"
        )
    if not (parsed.path or "").strip("/"):
        return


class _noop_trace:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _load_external_manifest(path_or_name: str) -> ExternalManifest:
    candidate = external_manifest_path(path_or_name)
    if not candidate.exists():
        util.die(
            f"external manifest '{path_or_name}' not found. Expected file path or "
            f"{settings.EXPERIMENTS_DIR / 'external' / (path_or_name + '.yaml')}"
        )
    raw = yaml.safe_load(candidate.read_text()) or {}
    name = str(raw.get("name") or candidate.stem)
    model = str(raw.get("model") or "")
    endpoint = str(raw.get("endpoint") or "")
    if not model or not endpoint:
        util.die(f"{candidate}: requires keys 'model' and 'endpoint'")
    _validate_endpoint(endpoint)
    runs = [str(tag) for tag in (raw.get("runs") or [])]
    workloads = [str(w) for w in (raw.get("workloads") or ["rate", "concurrent"])]
    request_model = str(
        raw.get("request_model")
        or raw.get("served_model")
        or raw.get("model_id")
        or ""
    ).strip()
    return ExternalManifest(
        name=name,
        model=model,
        endpoint=endpoint,
        api_key=str(raw.get("apikey") or raw.get("api_key") or ""),
        request_model=request_model,
        workloads=workloads,
        runs=runs,
    )


def external_manifest_path(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate

    direct = settings.EXPERIMENTS_DIR / "external" / f"{path_or_name}.yaml"
    if direct.exists():
        return direct

    if path_or_name.startswith("external/"):
        relative = settings.EXPERIMENTS_DIR / f"{path_or_name}.yaml"
        if relative.exists():
            return relative
    return candidate


def is_external_manifest(path: Path) -> bool:
    """True when a manifest lives under experiments/external/."""
    try:
        return path.resolve().is_relative_to((settings.EXPERIMENTS_DIR / "external").resolve())
    except (OSError, ValueError):
        return False
