"""Everything that talks to Kubernetes.

This is the whole of the old run.sh, in one place. Folding it into Python removed
the duplicated `sanitize()` (bash and Python had to agree because tags become
object names) and the shell/Python handoff where a benchmark failure had to be
smuggled back through an exit code.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import yaml

from . import engine, parallelism, render, util
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


def _gpu_request(container: dict[str, Any]) -> int:
    """GPUs one container holds. Extended resources can be declared either side."""
    resources = container.get("resources", {}) or {}
    for section in ("requests", "limits"):
        value = (resources.get(section) or {}).get("nvidia.com/gpu")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 1
    return 0


class Kube:
    def __init__(self, cluster: Cluster, *, dry_run: bool = False) -> None:
        self.cluster = cluster
        self.dry_run = dry_run
        if not dry_run and not shutil.which("kubectl"):
            util.die("kubectl not found on PATH")

    # -- plumbing ---------------------------------------------------------
    def _run(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        timeout: int | None = 300,
    ) -> subprocess.CompletedProcess[str]:
        """Run kubectl and hand back the whole result, stderr included.

        `_kubectl` yields stdout alone, which is what callers that parse output
        want but throws away *why* a command failed: a failed `exec` prints
        nothing on stdout, so its entire diagnostic lives in stderr.
        """
        cmd = ["kubectl", "-n", self.cluster.namespace, *args]
        try:
            return subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise KubectlError(f"timed out after {timeout}s: {' '.join(cmd)}") from None

    def _kubectl(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        check: bool = True,
        timeout: int | None = 300,
    ) -> str:
        proc = self._run(args, stdin=stdin, timeout=timeout)
        if check and proc.returncode != 0:
            cmd = ["kubectl", "-n", self.cluster.namespace, *args]
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
        self.apply(render.results_pvc(self.cluster, pvc_name=self.cluster.results_pvc))
        if self.cluster.external_results_pvc != self.cluster.results_pvc:
            self.apply(
                render.results_pvc(
                    self.cluster, pvc_name=self.cluster.external_results_pvc
                )
            )

    def apply_params(self, run: Run) -> str:
        manifest = render.params_configmap(self.cluster, run)
        self.apply(manifest)
        return manifest

    def apply_workloads(self, workloads: list[Workload]) -> None:
        self.apply(render.workloads_configmap(self.cluster, workloads))

    def host_facts(self, node_name: str) -> dict[str, Any]:
        """Which machine a run actually landed on.

        A sweep only means anything if consecutive runs share a substrate, and
        these nodes get recycled and driver-upgraded underneath us. Recording the
        host makes an environment change visible as an environment change,
        instead of looking like a regression in whatever config was being swept.
        """
        if not node_name:
            return {}
        try:
            node = self.get_json("get", "node", node_name)
        except KubectlError:
            return {}
        labels = node.get("metadata", {}).get("labels", {}) or {}
        facts: dict[str, Any] = {"node": node_name}
        for key, label in (
            ("gpu_product", "nvidia.com/gpu.product"),
            ("gpu_memory_mib", "nvidia.com/gpu.memory"),
            ("gpu_count", "nvidia.com/gpu.count"),
            ("gpu_driver", "nvidia.com/cuda.driver-version.full"),
        ):
            if label in labels:
                facts[key] = labels[label]
        return facts

    # -- sharing the GPUs -------------------------------------------------
    def foreign_gpu_holders(self) -> list[tuple[str, int]]:
        """(description, GPU count) for every live pod holding GPUs but ours."""
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
            held = sum(
                _gpu_request(container)
                for container in pod.get("spec", {}).get("containers", [])
            )
            if held:
                holders.append((f"{name} ({phase}, {held} GPU)", held))
        return holders

    def free_gpus(self) -> tuple[int, list[str]]:
        """How many GPUs are available to us, and who is sitting on the rest.

        Our own engine pod does not count against us: it is replaced, not joined,
        so with strategy Recreate its devices come back before the new pod starts.
        """
        holders = self.foreign_gpu_holders()
        taken = sum(count for _, count in holders)
        return max(0, self.cluster.gpus_per_node - taken), [name for name, _ in holders]

    def ensure_gpus(self, needed: int, timeout_s: int) -> bool:
        """Get `needed` GPUs free, scaling down known rivals only if we must.

        A two-GPU node makes "is the GPU free" the wrong question: a colleague's
        single-GPU job can coexist with a single-GPU run of ours, and scaling it
        down to take a device we do not need is just rude. Rivals therefore get
        scaled down when the arithmetic says we are short, and not before.
        """
        free, holders = self.free_gpus()
        if free >= needed:
            if holders:
                util.step(
                    f"{free} of {self.cluster.gpus_per_node} GPU(s) free, enough for this "
                    f"config; leaving {', '.join(holders)} alone"
                )
            return True

        util.step(
            f"need {needed} GPU(s), {free} free — scaling down known engines: "
            f"{', '.join(r.get('name', '?') for r in self.cluster.gpu_rivals) or '(none)'}"
        )
        self.scale_down_rivals()

        deadline = time.time() + timeout_s
        warned = False
        while True:
            free, holders = self.free_gpus()
            if free >= needed:
                return True
            if time.time() >= deadline:
                util.warn(
                    f"still only {free} of {needed} GPU(s) free after "
                    f"{util.human_duration(timeout_s)} (held by {', '.join(holders)}); "
                    f"deploying anyway"
                )
                return False
            if not warned:
                util.step(
                    f"waiting up to {util.human_duration(timeout_s)} for "
                    f"{', '.join(holders)} to release {needed - free} more GPU(s)"
                )
                warned = True
            time.sleep(20)

    def scale_down_rivals(self) -> None:
        """Scale the other engines sharing these GPUs to zero and wait for them."""
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

    # -- the engine -------------------------------------------------------
    def server_pod(self, *, exclude_uid: str | None = None) -> dict[str, Any] | None:
        pods = self.get_json(
            "get", "pods", "-l", f"app={self.cluster.app_label}"
        ).get("items", [])
        if exclude_uid:
            pods = [p for p in pods if p.get("metadata", {}).get("uid") != exclude_uid]
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
        # `rollout restart` returns before the replacement pod exists, and with
        # strategy: Recreate the outgoing pod stays Ready until it is torn down.
        # Without excluding it by identity the loop below matches the pod still
        # serving the *previous* run and reports "Ready in 0s" against a config
        # that was never applied.
        outgoing = self.server_pod()
        outgoing_uid = (outgoing or {}).get("metadata", {}).get("uid")

        self.apply(render.deployment(self.cluster, model, run))
        self._kubectl(["rollout", "restart", f"deployment/{self.cluster.deployment}"])

        deadline = time.time() + timeout_s
        started = time.time()
        last_note = ""
        restarts_seen = 0
        while time.time() < deadline:
            pod = self.server_pod(exclude_uid=outgoing_uid)
            if pod is None:
                if last_note != "replacing":
                    util.step(f"waiting for {self.cluster.deployment}: replacing pod")
                    last_note = "replacing"
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
                facts = engine.facts(log)
                facts.update(self.host_facts(pod.get("spec", {}).get("nodeName", "")))
                return DeployOutcome(True, "ready", log=log, facts=facts)

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

        pod = self.server_pod(exclude_uid=outgoing_uid)
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
            proc = self._run(
                ["exec", name, "--", "python3", "-c", script], timeout=timeout_s
            )
        except KubectlError as exc:
            return False, str(exc)
        out = (proc.stdout or "").strip()
        if "OK" in out:
            return True, out.splitlines()[-1]
        # The failure is almost always on stderr: an assertion from the script
        # above, or kubectl itself reporting that the container went away.
        detail = "\n".join(t for t in (out, (proc.stderr or "").strip()) if t)
        if not detail:
            return False, f"exec into {name} exited {proc.returncode} with no output"
        return False, detail.splitlines()[-1]

    @contextlib.contextmanager
    def engine_trace(self, log_path: Path) -> Iterator[None]:
        """Record the engine's own log for as long as the block runs.

        `server.log` is captured the instant the pod reports Ready, so it stops
        before any load arrives. When a benchmark stalls rather than errors, the
        evidence — KV cache usage, queue depth, prefill throughput — exists only
        in the engine log *during* the run, and was being thrown away.
        """
        pod = self.server_pod()
        if pod is None:
            yield
            return
        name = pod.get("metadata", {}).get("name", "")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sink = open(log_path, "a", encoding="utf-8")
        sink.write(f"\n===== engine trace: {name} =====\n")
        sink.flush()
        proc = subprocess.Popen(
            ["kubectl", "-n", self.cluster.namespace, "logs", "-f", name, "--since=1s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                sink.write(line)
                # Flushed per line: this trace exists to explain runs that get
                # killed, so buffered output is output we never see.
                sink.flush()

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            thread.join(timeout=15)
            sink.close()

    # -- benchmark jobs ---------------------------------------------------
    def reap_stale_bench_pods(self, run_tag: str) -> None:
        """Clear only this run's non-terminal benchmark pods.

        Different sweeps can run in parallel; deleting every live `app=guidellm`
        pod would kill another run's in-flight workload.
        """
        pods = self.get_json(
            "get", "pods", "-l", f"app=guidellm,bench.tag={run_tag}"
        ).get("items", [])
        stale = [
            p["metadata"]["name"]
            for p in pods
            if p.get("status", {}).get("phase") not in ("Succeeded", "Failed")
        ]
        for name in stale:
            util.warn(f"force-deleting stale benchmark pod {name}")
            self.delete("pod", name, force=True, wait=False)

    def run_benchmark(
        self,
        run: Run,
        workload: Workload,
        log_path: Path,
        *,
        echo: bool = True,
        service_url: str | None = None,
        api_key: str | None = None,
    ) -> JobOutcome:
        name = render.job_name(run, workload)
        manifest = render.benchmark_job(
            self.cluster, run, workload, service_url=service_url, api_key=api_key
        )
        util.step(f"workload '{workload.name}' -> job/{name}")
        self.reap_stale_bench_pods(run.tag)
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
        deadline = render.job_deadline(self.cluster, workload)
        return self._await_job(name, grace_s=deadline + 600, deadline_s=deadline)

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

    def _await_job(
        self,
        name: str,
        *,
        poll_s: int = 10,
        grace_s: int = 600,
        deadline_s: int | None = None,
        reason_wait_s: int = 60,
    ) -> JobOutcome:
        """Wait for the Job to reach a terminal state, and say why it got there.

        `status.failed` flips as soon as the pod fails, but the Job-level
        `Failed` condition carrying the reason (DeadlineExceeded,
        BackoffLimitExceeded) is written a moment later. Returning on `failed`
        alone therefore reported every failure as an unexplained one — which is
        how a sweep killed by its own activeDeadlineSeconds looked identical to
        a benchmark that crashed.
        """
        deadline = time.time() + grace_s
        reason_deadline: float | None = None
        while time.time() < deadline:
            status = self.get_json("get", "job", name).get("status", {}) or {}
            if int(status.get("succeeded") or 0) > 0:
                return JobOutcome(True, "complete")
            if int(status.get("failed") or 0) > 0:
                explained = self._job_failure(status, deadline_s)
                if explained:
                    return JobOutcome(False, "bench", detail=explained)
                if reason_deadline is None:
                    reason_deadline = time.time() + reason_wait_s
                elif time.time() >= reason_deadline:
                    return JobOutcome(
                        False, "bench", detail="benchmark Job failed (no reason reported)"
                    )
            time.sleep(poll_s)
        return JobOutcome(False, "bench", detail="Job neither completed nor failed in time")

    @staticmethod
    def _job_failure(status: dict[str, Any], deadline_s: int | None) -> str:
        """The Failed condition's reason and message, once Kubernetes writes it."""
        for cond in status.get("conditions") or []:
            if cond.get("type") != "Failed" or cond.get("status") != "True":
                continue
            reason = cond.get("reason") or ""
            message = cond.get("message") or ""
            detail = ": ".join(part for part in (reason, message) if part)
            if reason == "DeadlineExceeded" and deadline_s:
                detail += (
                    f" — killed at its {util.human_duration(deadline_s)} ceiling, so it "
                    f"wrote no results; the workload made no progress or needs longer"
                )
            return detail or "benchmark Job failed"
        return ""

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

    def pull(
        self, glob: str, dest: Path, *, quiet: bool = False, pvc: str | None = None
    ) -> list[str]:
        """Copy result files off the results PVC via a throwaway reader pod."""
        pod = "bench-reader"
        dest.mkdir(parents=True, exist_ok=True)
        self._utility_pod(
            pod, pvc or self.cluster.results_pvc, "/data", read_only=True
        )
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
            out = self._kubectl(
                [
                    "exec",
                    pod,
                    "--",
                    "sh",
                    "-c",
                    f"[ -d {hub} ] && ls {hub}/snapshots 2>/dev/null | head -1 || true",
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
        ours = 0
        for pod in pods:
            meta, status = pod.get("metadata", {}), pod.get("status", {})
            containers = status.get("containerStatuses") or [{}]
            held = sum(_gpu_request(c) for c in pod.get("spec", {}).get("containers", []))
            ours += held
            lines.append(
                f"  pod {meta.get('name')}: {status.get('phase')} "
                f"gpus={held} restarts={containers[0].get('restartCount', 0)} "
                f"node={pod.get('spec', {}).get('nodeName', '?')}"
            )
        cm = self.get_json("get", "configmap", self.cluster.params_configmap)
        if cm:
            labels = cm.get("metadata", {}).get("labels", {}) or {}
            data = cm.get("data", {}) or {}
            lines.append(
                f"live config: tag={labels.get('bench.tag', '?')} "
                f"model={data.get('SERVED_MODEL_NAME', '?')} "
                f"topology={parallelism.of(data).label} "
                f"mns={data.get('MAX_NUM_SEQS', '?')} mnbt={data.get('MAX_NUM_BATCHED_TOKENS', '?')}"
            )
        # Two different counts, because they answer different questions: what is
        # idle right now, and what the next run can have. Our own engine's devices
        # belong in the second — a redeploy takes the pod down before it starts the
        # replacement — which is why they are named separately rather than summed.
        free, holders = self.free_gpus()
        lines.append(
            f"GPUs: {self.cluster.gpus_per_node} on the node, {free} available to the "
            f"next run"
            + (f" ({ours} held by our engine, released on redeploy)" if ours else "")
            + (f", {', '.join(holders)} held by others" if holders else "")
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
