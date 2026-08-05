"""Everything that talks to Kubernetes.

This is the whole of the old run.sh, in one place. Folding it into Python removed
the duplicated `sanitize()` (bash and Python had to agree because tags become
object names) and the shell/Python handoff where a benchmark failure had to be
smuggled back through an exit code.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from . import engine, render, util
from .model_registry import Model
from .plan import Run
from .settings import Cluster
from .workload_registry import Workload


class KubectlError(RuntimeError):
    pass


@dataclasses.dataclass
class DeployOutcome:
    ok: bool
    reason: str
    detail: str = ""
    log: str = ""
    facts: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class JobOutcome:
    ok: bool
    reason: str
    detail: str = ""


class Kube:
    def __init__(self, cluster: Cluster, *, dry_run: bool = False) -> None:
        self.cluster = cluster
        self.dry_run = dry_run
        if not dry_run and not shutil.which("kubectl"):
            util.die("kubectl not found on PATH")

    # -- plumbing ---------------------------------------------------------
    def _kubectl(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        check: bool = True,
        timeout: int | None = 300,
    ) -> str:
        cmd = ["kubectl", "-n", self.cluster.namespace, *args]
        try:
            proc = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise KubectlError(f"timed out after {timeout}s: {' '.join(cmd)}") from None
        if check and proc.returncode != 0:
            raise KubectlError(
                f"{' '.join(cmd)} failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}"
            )
        return proc.stdout

    def _kubectl_cluster(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        check: bool = True,
        timeout: int | None = 300,
    ) -> str:
        """Run kubectl without forcing a namespace."""
        cmd = ["kubectl", *args]
        try:
            proc = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise KubectlError(f"timed out after {timeout}s: {' '.join(cmd)}") from None
        if check and proc.returncode != 0:
            raise KubectlError(
                f"{' '.join(cmd)} failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}"
            )
        return proc.stdout

    def apply(self, manifest: str, *, timeout: int = 300) -> None:
        self._kubectl(["apply", "-f", "-"], stdin=manifest, timeout=timeout)

    def get_json(self, *args: str) -> dict[str, Any]:
        out = self._kubectl([*args, "-o", "json"], check=False)
        if not out.strip():
            return {}
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {}

    def delete(self, kind: str, name: str, *, force: bool = False, wait: bool = True) -> None:
        args = ["delete", kind, name, "--ignore-not-found", f"--wait={'true' if wait else 'false'}"]
        if force:
            args += ["--grace-period=0", "--force"]
        self._kubectl(args, check=False, timeout=300)

    # -- prerequisites ----------------------------------------------------
    def ensure_namespace(self) -> None:
        """Create the target namespace when it does not already exist."""
        ns = self.cluster.namespace
        existing = self.get_json("get", "namespace", ns)
        if existing:
            return
        util.step(f"creating namespace {ns}")
        out = self._kubectl_cluster(["create", "namespace", ns], check=False, timeout=60)
        if "created" in out or "AlreadyExists" in out:
            return
        # One retry with strict checking surfaces the real error message.
        self._kubectl_cluster(["create", "namespace", ns], check=True, timeout=60)

    def ensure_secret(self) -> None:
        token = os.environ.get("HF_TOKEN")
        name = self.cluster.hf_secret
        if token:
            util.step(f"refreshing Secret {name} from $HF_TOKEN")
            manifest = self._kubectl(
                [
                    "create",
                    "secret",
                    "generic",
                    name,
                    f"--from-literal=HF_TOKEN={token}",
                    "--dry-run=client",
                    "-o",
                    "yaml",
                ]
            )
            self.apply(manifest)
            return
        existing = self.get_json("get", "secret", name)
        if not existing:
            util.die(
                f"Secret {name} is missing and $HF_TOKEN is not set.\n"
                f"    export HF_TOKEN=hf_xxx  and retry"
            )

    def ensure_results_pvc(self) -> None:
        self.apply(render.results_pvc(self.cluster))

    def ensure_model_cache_pvc(self, model: Model) -> None:
        self.apply(render.model_cache_pvc(self.cluster, model))

    def apply_params(self, run: Run) -> str:
        manifest = render.params_configmap(self.cluster, run)
        self.apply(manifest)
        return manifest

    def apply_workloads(self, workloads: list[Workload]) -> None:
        self.apply(render.workloads_configmap(self.cluster, workloads))

    # -- the single GPU ---------------------------------------------------
    def foreign_gpu_holders(self) -> list[str]:
        """Pods holding the GPU that aren't our engine."""
        pods = self.get_json("get", "pods").get("items", [])
        holders = []
        for pod in pods:
            meta = pod.get("metadata", {})
            name = meta.get("name", "")
            labels = meta.get("labels", {}) or {}
            if labels.get("app") == self.cluster.app_label:
                continue
            if name.startswith(self.cluster.deployment):
                continue
            phase = pod.get("status", {}).get("phase", "?")
            if phase in ("Succeeded", "Failed"):
                continue
            for container in pod.get("spec", {}).get("containers", []):
                requests = (container.get("resources", {}) or {}).get("requests", {}) or {}
                if "nvidia.com/gpu" in requests:
                    holders.append(f"{name} ({phase})")
                    break
        return holders

    def wait_for_free_gpu(self, timeout_s: int) -> bool:
        deadline = time.time() + timeout_s
        warned = False
        while True:
            holders = self.foreign_gpu_holders()
            if not holders:
                return True
            if time.time() >= deadline:
                util.warn(
                    f"GPU still held by {', '.join(holders)} after "
                    f"{util.human_duration(timeout_s)}; deploying anyway"
                )
                return False
            if not warned:
                util.step(
                    f"GPU held by another workload ({', '.join(holders)}); waiting up to "
                    f"{util.human_duration(timeout_s)} — this is a shared single-GPU cluster"
                )
                warned = True
            time.sleep(20)

    def free_gpu(self) -> None:
        """Scale down the other engines that share this GPU."""
        for rival in self.cluster.gpu_rivals:
            kind, name = rival.get("kind", "deployment"), rival.get("name")
            if not name:
                continue
            self._kubectl(["scale", kind, name, "--replicas=0"], check=False, timeout=60)
        for rival in self.cluster.gpu_rivals:
            name = rival.get("name")
            if name:
                self._kubectl(
                    ["wait", "--for=delete", "pod", "-l", f"app={name}", "--timeout=180s"],
                    check=False,
                    timeout=200,
                )

    def teardown_server(self, *, timeout_s: int = 300, force_after_s: int = 90) -> None:
        """Reset our own server to a clean slate before a fresh deploy."""
        util.step(f"tearing down any existing {self.cluster.deployment} pod(s)")
        # Best effort: if the Deployment exists, scale it to zero first so the
        # controller stops creating replacements while we clean up.
        self._kubectl(
            ["scale", "deployment", self.cluster.deployment, "--replicas=0"],
            check=False,
            timeout=60,
        )

        deadline = time.time() + timeout_s
        force_deadline = time.time() + force_after_s
        forced: set[str] = set()
        while time.time() < deadline:
            pods = self.get_json("get", "pods", "-l", f"app={self.cluster.app_label}").get(
                "items", []
            )
            if not pods:
                return

            now = time.time()
            for pod in pods:
                name = pod.get("metadata", {}).get("name", "")
                if not name:
                    continue
                deleting = bool(pod.get("metadata", {}).get("deletionTimestamp"))
                if now >= force_deadline or deleting:
                    if name not in forced:
                        util.warn(f"force-deleting stale server pod {name}")
                        forced.add(name)
                    self.delete("pod", name, force=True, wait=False)
                else:
                    self.delete("pod", name, force=False, wait=False)
            time.sleep(3)

        leftovers = [
            p.get("metadata", {}).get("name", "?")
            for p in self.get_json("get", "pods", "-l", f"app={self.cluster.app_label}").get(
                "items", []
            )
        ]
        if leftovers:
            raise KubectlError(
                f"could not fully tear down {self.cluster.deployment} pods within "
                f"{util.human_duration(timeout_s)}: {', '.join(leftovers)}"
            )

    # -- the engine -------------------------------------------------------
    def server_pod(self) -> dict[str, Any] | None:
        pods = self.get_json(
            "get", "pods", "-l", f"app={self.cluster.app_label}"
        ).get("items", [])
        if not pods:
            return None
        return sorted(
            pods, key=lambda p: p.get("metadata", {}).get("creationTimestamp", "")
        )[-1]

    def pod_logs(self, name: str, *, previous: bool = False, tail_lines: int = 4000) -> str:
        args = ["logs", name, f"--tail={tail_lines}"]
        if previous:
            args.append("--previous")
        try:
            return self._kubectl(args, check=False, timeout=120)
        except KubectlError:
            return ""

    def deploy(self, run: Run, model: Model, *, timeout_s: int) -> DeployOutcome:
        """Apply this run's config and restart the server, watching for a crash.

        `kubectl rollout status` alone would sit here for the full timeout while
        a pod that OOMed at startup crash-loops, so this polls the pod and bails
        out as soon as the log or the container state says it will never be Ready.
        """
        self.teardown_server()
        reclaim_delay_s = 20
        util.step(
            f"waiting {reclaim_delay_s}s for GPU memory to settle after teardown"
        )
        time.sleep(reclaim_delay_s)
        self.apply(render.deployment(self.cluster, model, run))
        self._kubectl(["rollout", "restart", f"deployment/{self.cluster.deployment}"])

        deadline = time.time() + timeout_s
        started = time.time()
        last_note = ""
        restarts_seen = 0
        while time.time() < deadline:
            pod = self.server_pod()
            if pod is None:
                time.sleep(5)
                continue
            name = pod.get("metadata", {}).get("name", "")
            status = pod.get("status", {}) or {}
            phase = status.get("phase", "?")
            containers = status.get("containerStatuses") or []
            state = (containers[0].get("state") if containers else {}) or {}
            last_state = (containers[0].get("lastState") if containers else {}) or {}
            restarts = int(containers[0].get("restartCount", 0)) if containers else 0

            if containers and containers[0].get("ready"):
                log = self.pod_logs(name)
                elapsed = time.time() - started
                util.ok(f"server Ready in {util.human_duration(elapsed)} ({name})")
                return DeployOutcome(True, "ready", log=log, facts=engine.facts(log))

            waiting = (state.get("waiting") or {}).get("reason", "")
            if waiting in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull",
                           "CreateContainerConfigError", "CreateContainerError"):
                log = self.pod_logs(name, previous=True) or self.pod_logs(name)
                kind, detail = engine.label_failure(log)
                return DeployOutcome(False, kind if waiting == "CrashLoopBackOff" else "startup",
                                     detail=detail or waiting, log=log,
                                     facts=engine.facts(log))

            terminated = (last_state.get("terminated") or {})
            if restarts > restarts_seen and terminated:
                restarts_seen = restarts
                log = self.pod_logs(name, previous=True) or self.pod_logs(name)
                kind, detail = engine.label_failure(log)
                return DeployOutcome(False, kind, detail=detail, log=log,
                                     facts=engine.facts(log))

            if phase == "Failed":
                log = self.pod_logs(name)
                kind, detail = engine.label_failure(log)
                return DeployOutcome(False, kind, detail=detail, log=log,
                                     facts=engine.facts(log))

            # Running but not Ready: the model may still be loading, or the
            # engine may have already logged something fatal and hung.
            log = self.pod_logs(name, tail_lines=400)
            fatal = engine.classify_failure(log)
            if fatal:
                full = self.pod_logs(name)
                return DeployOutcome(False, fatal[0], detail=fatal[1], log=full,
                                     facts=engine.facts(full))

            note = f"{phase}/{waiting or 'loading'}"
            if note != last_note:
                util.step(f"waiting for {self.cluster.deployment}: {note}")
                last_note = note
            time.sleep(10)

        pod = self.server_pod()
        log = self.pod_logs(pod["metadata"]["name"]) if pod else ""
        return DeployOutcome(
            False,
            "timeout",
            detail=f"not Ready within {util.human_duration(timeout_s)}",
            log=log,
            facts=engine.facts(log),
        )

    def smoke_test(self, model: Model, *, timeout_s: int = 120) -> tuple[bool, str]:
        """Confirm the server actually serves this model before we benchmark it.

        A Ready /health check only proves the HTTP server is up. This catches the
        case where the engine is serving something other than what we asked for,
        or cannot complete a single request — before an hour of load testing
        produces a directory full of errors.
        """
        pod = self.server_pod()
        if pod is None:
            return False, "no server pod"
        name = pod["metadata"]["name"]
        script = (
            "import json,urllib.request\n"
            "base='http://localhost:8000'\n"
            "models=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))\n"
            "served=[m['id'] for m in models.get('data',[])]\n"
            f"want={model.served_name!r}\n"
            "assert want in served, 'serving %s, expected %s' % (served, want)\n"
            "body=json.dumps({'model':want,'messages':[{'role':'user','content':'ping'}],"
            "'max_tokens':1,'temperature':0}).encode()\n"
            "req=urllib.request.Request(base+'/v1/chat/completions',data=body,"
            "headers={'Content-Type':'application/json'})\n"
            "out=json.load(urllib.request.urlopen(req,timeout=90))\n"
            "assert out['choices'][0]['message'] is not None, out\n"
            "print('OK', served)\n"
        )
        try:
            out = self._kubectl(
                ["exec", name, "--", "python3", "-c", script],
                check=True,
                timeout=timeout_s,
            )
        except KubectlError as exc:
            return False, str(exc)
        if "OK" in out:
            return True, out.strip().splitlines()[-1]
        return False, (out.strip() or "smoke test produced no output").splitlines()[-1]

    # -- benchmark jobs ---------------------------------------------------
    def reap_stale_bench_pods(self) -> None:
        pods = self.get_json("get", "pods", "-l", "app=guidellm").get("items", [])
        stale = [
            p["metadata"]["name"]
            for p in pods
            if p.get("status", {}).get("phase") not in ("Succeeded", "Failed")
        ]
        for name in stale:
            util.warn(f"force-deleting stale benchmark pod {name}")
            self.delete("pod", name, force=True, wait=False)

    def run_benchmark(
        self, run: Run, workload: Workload, log_path: Path, *, echo: bool = True
    ) -> JobOutcome:
        name = render.job_name(run, workload)
        manifest = render.benchmark_job(self.cluster, run, workload)
        util.step(f"workload '{workload.name}' -> job/{name}")
        self.reap_stale_bench_pods()
        self.delete("job", name, wait=True)
        self.apply(manifest)

        start_timeout = self.cluster.timeout("job_start_s", 300)
        self._kubectl(
            [
                "wait",
                "--for=condition=Ready",
                "pod",
                "-l",
                f"job-name={name}",
                f"--timeout={start_timeout}s",
            ],
            check=False,
            timeout=start_timeout + 30,
        )
        self._stream_logs(f"job/{name}", log_path, echo=echo)
        return self._await_job(name)

    def _stream_logs(self, target: str, log_path: Path | None, *, echo: bool) -> None:
        cmd = ["kubectl", "-n", self.cluster.namespace, "logs", "-f", target]
        sink = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sink = open(log_path, "a", encoding="utf-8")
            sink.write(f"\n===== {' '.join(cmd)} =====\n")
            sink.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if sink is not None:
                    sink.write(line)
                if echo:
                    print(f"    {line.rstrip()}", flush=True)
        finally:
            proc.wait()
            if sink is not None:
                sink.close()

    def _await_job(self, name: str, *, poll_s: int = 10, grace_s: int = 600) -> JobOutcome:
        deadline = time.time() + grace_s
        while time.time() < deadline:
            job = self.get_json("get", "job", name)
            status = job.get("status", {}) or {}
            if int(status.get("succeeded") or 0) > 0:
                return JobOutcome(True, "complete")
            if int(status.get("failed") or 0) > 0:
                reason = ""
                for cond in status.get("conditions") or []:
                    if cond.get("type") == "Failed":
                        reason = cond.get("reason") or cond.get("message") or ""
                return JobOutcome(False, "bench", detail=reason or "benchmark Job failed")
            time.sleep(poll_s)
        return JobOutcome(False, "bench", detail="Job neither completed nor failed in time")

    # -- results ----------------------------------------------------------
    def _utility_pod(
        self,
        name: str,
        pvc: str,
        mount: str,
        *,
        read_only: bool,
        node: str | None = None,
    ) -> None:
        """Start a throwaway busybox pod with `pvc` mounted, and wait for it."""
        spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "util",
                    "image": self.cluster.image("utility"),
                    "command": ["sleep", "3600"],
                    "volumeMounts": [
                        {"name": "vol", "mountPath": mount, "readOnly": read_only}
                    ],
                }
            ],
            "volumes": [{"name": "vol", "persistentVolumeClaim": {"claimName": pvc}}],
        }
        if node:
            spec["nodeSelector"] = {"kubernetes.io/hostname": node}
        manifest = yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": name,
                    "namespace": self.cluster.namespace,
                    "labels": {"app": name, "bench.suite": "vllm"},
                },
                "spec": spec,
            }
        )
        self.delete("pod", name, wait=True)
        self.apply(manifest)
        self._kubectl(
            ["wait", "--for=condition=Ready", f"pod/{name}", "--timeout=180s"], timeout=200
        )

    def pull(self, glob: str, dest: Path, *, quiet: bool = False) -> list[str]:
        """Copy result files off the results PVC via a throwaway reader pod."""
        pod = "bench-reader"
        dest.mkdir(parents=True, exist_ok=True)
        self._utility_pod(pod, self.cluster.results_pvc, "/data", read_only=True)
        try:
            listing = self._kubectl(
                ["exec", pod, "--", "sh", "-c", f"ls /data/{glob} 2>/dev/null"], check=False
            )
            names = [line.strip() for line in listing.splitlines() if line.strip()]
            pulled = []
            for remote in names:
                local = dest / Path(remote).name
                data = subprocess.run(
                    [
                        "kubectl",
                        "-n",
                        self.cluster.namespace,
                        "exec",
                        pod,
                        "--",
                        "cat",
                        remote,
                    ],
                    capture_output=True,
                    timeout=600,
                )
                if data.returncode != 0 or not data.stdout:
                    util.warn(f"could not read {remote} off the PVC")
                    continue
                local.write_bytes(data.stdout)
                pulled.append(local.name)
                if not quiet:
                    util.step(f"pulled {local.name}")
            return pulled
        finally:
            self.delete("pod", pod, wait=False)

    def clean_results(self, glob: str) -> str:
        pod = "bench-cleaner"
        self._utility_pod(pod, self.cluster.results_pvc, "/data", read_only=False)
        try:
            return self._kubectl(
                ["exec", pod, "--", "sh", "-c", f"rm -fv /data/{glob}"], check=False
            )
        finally:
            self.delete("pod", pod, wait=False)

    # -- weights ----------------------------------------------------------
    def gpu_node(self) -> str | None:
        """The node our engine runs on, i.e. where the RWO weight PVC attaches."""
        pod = self.server_pod()
        node = (pod or {}).get("spec", {}).get("nodeName")
        if node:
            return node
        nodes = self.get_json("get", "nodes").get("items", [])
        for candidate in nodes:
            capacity = candidate.get("status", {}).get("capacity", {}) or {}
            if int(capacity.get("nvidia.com/gpu", 0) or 0) > 0:
                return candidate.get("metadata", {}).get("name")
        return None

    def model_is_cached(self, model: Model) -> bool | None:
        """True/False if we could check, None if the probe itself failed."""
        node = self.gpu_node()
        if not node:
            util.warn("could not identify the GPU node; skipping the weight-cache probe")
            return None
        pod = "bench-cache-probe"
        try:
            self._utility_pod(
                pod, model.cache_pvc, self.cluster.model_mount, read_only=True, node=node
            )
        except KubectlError as exc:
            util.warn(f"weight-cache probe could not start ({exc}); assuming weights present")
            self.delete("pod", pod, wait=False)
            return None
        try:
            hub = f"{self.cluster.model_mount}/huggingface/hub/{model.cache_dir_name}"
            revision = (model.revision or "").strip()
            if revision:
                # HF cache records named refs under refs/<name>, each pointing to
                # a snapshot hash directory under snapshots/<commit>.
                check_cmd = (
                    f"ref_file={hub}/refs/{revision}; "
                    f"[ -f \"$ref_file\" ] && ref=$(cat \"$ref_file\") && "
                    f"[ -d {hub}/snapshots/$ref ] && echo \"$ref\" || true"
                )
            else:
                check_cmd = f"[ -d {hub} ] && ls {hub}/snapshots 2>/dev/null | head -1 || true"
            out = self._kubectl(
                [
                    "exec",
                    pod,
                    "--",
                    "sh",
                    "-c",
                    check_cmd,
                ],
                check=False,
            )
            return bool(out.strip())
        finally:
            self.delete("pod", pod, wait=False)

    def preseed(self, model: Model) -> bool:
        node = self.gpu_node()
        if not node:
            util.warn("could not identify the GPU node; skipping pre-seed")
            return False
        name = f"preseed-{model.name}"[:52]
        util.log(f"pre-seeding {model.hf_repo} onto {model.cache_pvc} (node {node})")
        self.delete("job", name, wait=True)
        self.apply(render.preseed_job(self.cluster, model, node))
        start = self.cluster.timeout("preseed_s", 10800)
        self._kubectl(
            ["wait", "--for=condition=Ready", "pod", "-l", f"job-name={name}", "--timeout=300s"],
            check=False,
            timeout=330,
        )
        self._stream_logs(f"job/{name}", None, echo=True)
        outcome = self._await_job(name, grace_s=start)
        if outcome.ok:
            util.ok(f"weights for {model.name} are on {model.cache_pvc}")
        else:
            util.warn(f"pre-seed did not finish cleanly: {outcome.detail}")
        return outcome.ok

    # -- inspection -------------------------------------------------------
    def status_lines(self) -> list[str]:
        lines = []
        dep = self.get_json("get", "deployment", self.cluster.deployment)
        if dep:
            spec_replicas = dep.get("spec", {}).get("replicas", 0)
            ready = dep.get("status", {}).get("readyReplicas", 0) or 0
            lines.append(f"{self.cluster.deployment}: {ready}/{spec_replicas} ready")
        else:
            lines.append(f"{self.cluster.deployment}: not deployed")
        pods = self.get_json("get", "pods", "-l", f"app={self.cluster.app_label}").get("items", [])
        for pod in pods:
            meta, status = pod.get("metadata", {}), pod.get("status", {})
            containers = status.get("containerStatuses") or [{}]
            lines.append(
                f"  pod {meta.get('name')}: {status.get('phase')} "
                f"restarts={containers[0].get('restartCount', 0)} "
                f"node={pod.get('spec', {}).get('nodeName', '?')}"
            )
        cm = self.get_json("get", "configmap", self.cluster.params_configmap)
        if cm:
            labels = cm.get("metadata", {}).get("labels", {}) or {}
            data = cm.get("data", {}) or {}
            lines.append(
                f"live config: tag={labels.get('bench.tag', '?')} "
                f"model={data.get('SERVED_MODEL_NAME', '?')} "
                f"mns={data.get('MAX_NUM_SEQS', '?')} mnbt={data.get('MAX_NUM_BATCHED_TOKENS', '?')}"
            )
        return lines

    def bench_jobs(self) -> list[tuple[str, str, str]]:
        jobs = self.get_json("get", "jobs", "-l", "bench.suite=vllm").get("items", [])
        rows = []
        for job in sorted(
            jobs, key=lambda j: j.get("metadata", {}).get("creationTimestamp", "")
        ):
            meta, status = job.get("metadata", {}), job.get("status", {})
            state = (
                "succeeded"
                if status.get("succeeded")
                else "failed"
                if status.get("failed")
                else "running"
            )
            rows.append((meta.get("name", "?"), state, meta.get("creationTimestamp", "")))
        return rows

    def clean_jobs(self) -> str:
        return self._kubectl(
            ["delete", "job", "-l", "bench.suite=vllm", "--ignore-not-found"], check=False
        )
