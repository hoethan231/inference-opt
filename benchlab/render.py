"""Render the cluster/ manifest templates for a run.

Placeholder substitution used to be `sed s/__TAG__/.../` inside run.sh, which is
why the benchmark Jobs could only ever vary their tag. Doing it here means the
model's cache volume, the workload, the image and every timeout are all
substitutable, and `bench render` can show exactly what will be applied.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import yaml

from . import parallelism, settings
from .model_registry import Model
from .plan import Run
from .settings import Cluster
from .workload_registry import Workload

PLACEHOLDER_RE = re.compile(r"__[A-Z0-9_]+__")


def _substitute(template: Path, values: dict[str, str]) -> str:
    text = template.read_text()
    for name, value in values.items():
        text = text.replace(f"__{name}__", str(value))
    leftover = sorted(set(PLACEHOLDER_RE.findall(text)))
    if leftover:
        raise SystemExit(
            f"{template}: unsubstituted placeholder(s) {', '.join(leftover)} — "
            f"the template and benchlab/render.py have drifted apart"
        )
    return text


def _common(cluster: Cluster) -> dict[str, str]:
    return {
        "NAMESPACE": cluster.namespace,
        "DEPLOYMENT": cluster.deployment,
        "SERVICE": cluster.service,
        "APP_LABEL": cluster.app_label,
        "PARAMS_CONFIGMAP": cluster.params_configmap,
        "WORKLOADS_CONFIGMAP": cluster.workloads_configmap,
        "HF_SECRET": cluster.hf_secret,
        "RESULTS_PVC": cluster.results_pvc,
        "EXTERNAL_RESULTS_PVC": cluster.external_results_pvc,
        "RESULTS_MOUNT": cluster.results_mount,
        "MODEL_MOUNT": cluster.model_mount,
        "SERVICE_URL": cluster.service_url,
    }


def params_configmap(cluster: Cluster, run: Run) -> str:
    """The ConfigMap carrying this run's vLLM flags."""
    manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": cluster.params_configmap,
            "namespace": cluster.namespace,
            "labels": {
                "app": cluster.app_label,
                "bench.tag": run.tag[:63],
                "bench.experiment": run.experiment[:63],
                "bench.model": run.model[:63],
            },
        },
        "data": dict(sorted(run.params.items())),
    }
    # width: don't fold long flag strings across lines; they stay greppable.
    return yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False, width=4096)


def workloads_configmap(cluster: Cluster, workloads: list[Workload]) -> str:
    """The ConfigMap the benchmark Jobs mount their scenarios from."""
    data = {w.scenario_name: w.path.read_text() for w in workloads}
    manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": cluster.workloads_configmap,
            "namespace": cluster.namespace,
            "labels": {"app": cluster.app_label},
        },
        "data": data,
    }
    return yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False, width=4096)


def deployment(cluster: Cluster, model: Model, run: Run) -> str:
    """The engine Deployment for one config.

    Takes the run, not just the model, because the pod's GPU request is a
    function of the config's parallelism — the two cannot be allowed to disagree.
    """
    values = _common(cluster)
    values.update(
        {
            "VLLM_IMAGE": model.image or cluster.image("vllm"),
            "MODEL_CACHE_PVC": model.cache_pvc,
            "GPU_COUNT": str(parallelism.of(run.params).gpus),
        }
    )
    return _substitute(settings.DEPLOYMENT_TEMPLATE, values)


def results_pvc(cluster: Cluster, *, pvc_name: str | None = None) -> str:
    values = _common(cluster)
    if pvc_name:
        values["RESULTS_PVC"] = pvc_name
    return _substitute(settings.RESULTS_PVC_TEMPLATE, values)


def model_cache_pvc(cluster: Cluster, model: Model) -> str:
    values = _common(cluster)
    values.update(
        {
            "MODEL_CACHE_PVC": model.cache_pvc,
            "MODEL_CACHE_PVC_STORAGE": cluster.model_cache_pvc_storage,
        }
    )
    return _substitute(settings.MODEL_CACHE_PVC_TEMPLATE, values)


def job_name(run: Run, workload: Workload) -> str:
    return f"{run.tag}-{workload.code}"


def output_base(run: Run, workload: Workload) -> str:
    """Result filename stem on the results PVC (and in the run directory)."""
    return f"{run.tag}__{workload.name}"


def job_deadline(cluster: Cluster, workload: Workload) -> int:
    """Wall-clock ceiling for a benchmark Job.

    Headroom over the scenario's own duration so a slow warmup doesn't get
    killed as a false failure. The estimate counts the sending window, but a
    workload also has to drain whatever is still in flight when that window
    closes — with multi-turn conversations against a slow model that tail can
    be several times the window, so the slack is configurable per cluster
    rather than fixed to the one model it was first tuned against.
    """
    pct = cluster.timeout("bench_slack_pct", 350)
    floor = cluster.timeout("bench_slack_s", 900)
    return int(workload.est_seconds * pct / 100) + floor


def benchmark_job(
    cluster: Cluster,
    run: Run,
    workload: Workload,
    *,
    service_url: str | None = None,
    api_key: str | None = None,
) -> str:
    values = _common(cluster)
    external = run.strategy == "external"
    if service_url and not external:
        values["SERVICE_URL"] = service_url
    deadline = job_deadline(cluster, workload)
    values.update(
        {
            "JOB_NAME": job_name(run, workload),
            "GUIDELLM_IMAGE": cluster.image("guidellm"),
            "WORKLOAD_LABEL": workload.name[:63],
            "TAG_LABEL": run.tag[:63],
            "SCENARIO_FILE": workload.scenario_name,
            "OUT_BASE": output_base(run, workload),
            "DEADLINE": str(deadline),
            "MODEL_NAME": run.params.get("MODEL_NAME", run.model),
            "OPENAI_MODEL": run.params.get(
                "SERVED_MODEL_NAME", run.params.get("MODEL_NAME", run.model)
            ),
            "OPENAI_API_KEY": api_key or "",
        }
    )
    template = (
        settings.EXTERNAL_BENCH_JOB_TEMPLATE
        if external
        else settings.BENCH_JOB_TEMPLATE
    )
    if external:
        target, request_format = _external_backend(service_url or "")
        values["EXTERNAL_TARGET"] = target
        values["EXTERNAL_REQUEST_FORMAT"] = request_format
        values["EXTERNAL_FULL_ENDPOINT"] = f"{target.rstrip('/')}{request_format}"
    return _substitute(template, values)


def _external_backend(endpoint: str) -> tuple[str, str]:
    """Map an external endpoint to GuideLLM target/request_format fields.

    GuideLLM accepts canonical request formats like '/v1/chat/completions', so
    enterprise prefixes (for example '/enterpriseai') must live in `target`.
    """
    parsed = urlparse(endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1/chat/completions"):
        target = base + path[: -len("/v1/chat/completions")]
        return target.rstrip("/"), "/v1/chat/completions"
    if not path:
        return base, "/v1/chat/completions"
    # Endpoint is a base prefix like '/enterpriseai'.
    return base + path, "/v1/chat/completions"


def preseed_job(cluster: Cluster, model: Model, node: str) -> str:
    values = _common(cluster)
    values.update(
        {
            "JOB_NAME": f"preseed-{model.name}"[:52],
            "MODEL_LABEL": model.name[:63],
            "VLLM_IMAGE": model.image or cluster.image("vllm"),
            "HF_REPO": model.hf_repo,
            "HF_HOME": f"{cluster.model_mount}/huggingface",
            "MODEL_CACHE_PVC": model.cache_pvc,
            "NODE": node,
            "DEADLINE": str(cluster.timeout("preseed_s", 10800)),
        }
    )
    return _substitute(settings.PRESEED_JOB_TEMPLATE, values)
