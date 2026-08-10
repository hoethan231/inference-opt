"""Repo paths and the cluster wiring loaded from cluster/cluster.yaml."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent

CLUSTER_DIR = REPO / "cluster"
CLUSTER_YAML = CLUSTER_DIR / "cluster.yaml"
PARAMS_BASE_YAML = CLUSTER_DIR / "params-base.yaml"
DEPLOYMENT_TEMPLATE = CLUSTER_DIR / "deployments" / "vllm.yaml"
RESULTS_PVC_TEMPLATE = CLUSTER_DIR / "deployments" / "results-pvc.yaml"
BENCH_JOB_TEMPLATE = CLUSTER_DIR / "jobs" / "benchmark.yaml"
EXTERNAL_BENCH_JOB_TEMPLATE = CLUSTER_DIR / "jobs" / "benchmark-external.yaml"
PRESEED_JOB_TEMPLATE = CLUSTER_DIR / "jobs" / "preseed.yaml"

MODELS_DIR = REPO / "models"
EXPERIMENTS_DIR = REPO / "experiments"
WORKLOADS_DIR = REPO / "workloads"
WORKLOADS_INDEX = WORKLOADS_DIR / "index.yaml"

RUNS_DIR = REPO / "runs"
INDEX_PATH = RUNS_DIR / "index.jsonl"
REPORTS_DIR = REPO / "reports"

# Tags become Kubernetes object names: a Job is "<tag>-<workload code>" (code is
# capped at 6) and its pod appends "-xxxxx", all of which must fit in RFC-1123's
# 63 characters.
MAX_TAG_LEN = 50

# Rough per-run cost of deploying and waiting for a warm model load. Only used
# for the wall-clock estimate printed by dry runs.
DEPLOY_OVERHEAD_S = 300


@dataclasses.dataclass(frozen=True)
class Cluster:
    """The Kubernetes object names and knobs from cluster/cluster.yaml."""

    namespace: str
    deployment: str
    service: str
    app_label: str
    params_configmap: str
    workloads_configmap: str
    hf_secret: str
    results_pvc: str
    external_results_pvc: str
    results_mount: str
    model_mount: str
    gpus_per_node: int
    gpu_rivals: list[dict[str, str]]
    images: dict[str, str]
    timeouts: dict[str, int]

    @classmethod
    def load(cls, path: Path = CLUSTER_YAML) -> "Cluster":
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in raw]
        if missing:
            raise SystemExit(f"{path}: missing required key(s): {', '.join(missing)}")
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def service_url(self) -> str:
        return f"http://{self.service}.{self.namespace}.svc.cluster.local:8000"

    def image(self, key: str) -> str:
        try:
            return self.images[key]
        except KeyError:
            raise SystemExit(f"{CLUSTER_YAML}: images.{key} is not defined") from None

    def timeout(self, key: str, default: int) -> int:
        return int(self.timeouts.get(key, default))


def load_cluster() -> Cluster:
    return Cluster.load()
