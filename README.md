# vLLM hyperparameter benchmarking

A minimal, edit-one-file workflow for sweeping vLLM serving hyperparameters on
the cluster and benchmarking each config with GuideLLM.

The loop is four steps:

1. **Edit** `config/vllm-params.yaml` (change a hyperparameter).
2. **Deploy** — applies the config and restarts the server with it.
3. **Bench** — fires a load test at the server.
4. **Pull** — copies the result files back to your machine.

```bash
export HF_TOKEN=hf_xxx        # only needed the first time (creates the Secret)

./run.sh deploy               # apply params + (re)start server, wait until Ready
./run.sh bench both           # run the rate + concurrent-users benchmarks
./run.sh pull                 # copy result JSON/HTML into ./results/
```

Or in one go: `./run.sh all both`.

---

## The 30-second Kubernetes primer

Everything lives in the `vllm-inf` **namespace** (a folder that groups objects).
You never talk to a machine directly; you declare **objects** and Kubernetes
makes reality match them. The objects we use:

| Object | What it is | Our instance |
|---|---|---|
| **ConfigMap** | a bag of key/value settings injected into a pod as env vars | `vllm-params` (your knobs), `coding-agent-scenarios` (workloads) |
| **Secret** | like a ConfigMap but for sensitive values | `hf-token-secret` (your HF token) |
| **PVC** (PersistentVolumeClaim) | a network disk that outlives pods | `nemotron-model-cache` (weights), `guidellm-results-pvc` (results) |
| **Deployment** | keeps N copies of a pod running; restarts them on change | `vllm-server` (the vLLM engine) |
| **Service** | a stable DNS name/IP in front of pods | `vllm-svc` |
| **Job** | a pod that runs once to completion | the GuideLLM benchmark runs |

**Why a pod can't see your files:** a pod is an isolated container. It only
gets settings you inject (ConfigMap/Secret env vars) and disks you mount (PVCs).
That's why the token goes in a Secret and the weights live on a PVC.

**The single-GPU rule:** there is exactly **one** GPU in the cluster, and the
weights PVC is `ReadWriteOnce` (one pod at a time). So only one engine pod can
run at once. The Deployment uses `strategy: Recreate`, which tears the old pod
down *before* starting the new one — otherwise the new pod would wait forever
for a GPU the old pod still holds. `./run.sh deploy` also scales down the older
sample engines (`sglang-nemotron`, `vllm-nemotron`) as a safety net.

---

## Files

```
config/vllm-params.yaml     # <- EDIT THIS to tune. One key per hyperparameter.
deployments/vllm.yaml       # vLLM Deployment (reads the ConfigMap) + Service
jobs/vllm_rate.yaml         # benchmark Job: Poisson 20 req/s
jobs/vllm_user.yaml         # benchmark Job: 8/16/32/64 concurrent users
scenarios/*.json            # the GuideLLM workload definitions
run.sh                      # the driver (deploy | bench | all | pull | status | clean)
results/                    # local copies of pulled results (created by pull)
```

## How tuning actually works

`config/vllm-params.yaml` is a ConfigMap with one key per knob. The Deployment
injects all keys as env vars and a tiny shell entrypoint expands them into vLLM
flags, e.g. `MAX_NUM_SEQS: "128"` becomes `--max-num-seqs 128`. Because the
flags come from the ConfigMap, **you never edit the Deployment to re-tune** —
you edit the ConfigMap and run `./run.sh deploy`, which re-applies it and forces
a fresh pod so the change takes effect.

The knobs:

| Key | vLLM flag | Meaning |
|---|---|---|
| `MAX_MODEL_LEN` | `--max-model-len` | context window (tokens) |
| `MAX_NUM_SEQS` | `--max-num-seqs` | max concurrent running requests ("bouncer") |
| `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | prefill chunk size ("chewer") |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | KV-cache precision (`fp8` halves its size) |
| `GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | fraction of VRAM to use |
| `TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` | GPUs per replica (1 here) |
| `EXTRA_ARGS` | *(appended verbatim)* | any other flag, e.g. `--enforce-eager`, `--speculative-config '...'` |

Anything not promoted to its own key goes in `EXTRA_ARGS` — the escape hatch for
the long tail of vLLM options, no Deployment edit required.

## The two benchmarks

Both hit `/v1/chat/completions` with the same synthetic coding-agent data
(large shared prefixes, bursty prompt/output sizes, 20-turn tool-heavy
conversations). They differ only in how load is applied:

- **rate** (`jobs/vllm_rate.yaml`) — a steady 20 req/s Poisson stream; measures
  behavior under fixed offered load / saturation. ~5–6 min.
- **concurrent** (`jobs/vllm_user.yaml`) — fixed busy users at 8, 16, 32, 64
  streams; measures how latency/throughput scale with concurrency. ~20 min.

Each run is tagged (default: a timestamp) so result files never overwrite each
other: `vllm-rate-<tag>.json` / `vllm-users-<tag>.json` on the results PVC.
Pass your own tag to line runs up with a config: `./run.sh bench both ms128-pf2048`.

## Commands

```bash
./run.sh deploy                 # apply params + restart server, wait until Ready
./run.sh bench rate             # just the rate test
./run.sh bench concurrent tagA  # just the concurrent test, custom tag
./run.sh bench both             # both tests
./run.sh all both               # deploy then bench both
./run.sh pull [tag]             # copy results -> ./results/
./run.sh status                 # server + benchmark job status
./run.sh clean                  # delete this repo's benchmark jobs
```

## Prerequisites (already satisfied on this cluster)

- 1 GPU node (`nvidia.com/gpu: 1`, RTX PRO 6000 ~96 GB).
- Default StorageClass `nutanix-volume` (ReadWriteOnce).
- `nemotron-model-cache` PVC with the NVFP4 weights already downloaded — the
  server mounts it, so there is **no download step** and startup is fast.
- `guidellm-results-pvc` for results; `hf-token-secret` for the token.

## SGLang (later)

Adding SGLang is symmetric: a second Deployment (`strategy: Recreate`, same GPU
+ model-cache PVC) fronted by its own Service, plus `--model-path` /
`--chunked-prefill-size` / `--max-running-requests` flags instead of the vLLM
ones. Because of the single-GPU rule you'll deploy one engine at a time; the
benchmark Jobs just point at whichever Service is live.
