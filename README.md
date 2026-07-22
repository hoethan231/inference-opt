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
| **PVC** (PersistentVolumeClaim) | a network disk that outlives pods | `nemotron-model-cache` (weights), `vllm-sweep-results-pvc` (results, this repo's own) |
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
| `BLOCK_SIZE` | `--block-size` | KV-cache block size (tokens/block) |
| `VLLM_ATTENTION_BACKEND` | `--attention-backend` *(turned into the flag by the entrypoint — vLLM ≥0.13 no longer reads this as an env var directly, see [vllm-project/vllm#26315](https://github.com/vllm-project/vllm/pull/26315))* | attention kernel backend; empty = auto-select |
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
- `vllm-sweep-results-pvc` (20Gi, created automatically by `./run.sh deploy`
  from `deployments/results-pvc.yaml`) for results — dedicated to this repo's
  own `vllm-rate-*`/`vllm-users-*` jobs so it can't fill up from, or get
  cleared alongside, other benchmarking work on the shared cluster. If it
  ever does fill up, `./run.sh pull` then `./run.sh clean-results` clears it
  safely; **never** point these jobs back at the old shared
  `guidellm-results-pvc`.
- `hf-token-secret` for the token.

## Grid-searching hyperparameters (`sweep/`)

`sweep/` automates the tune-loop above across many configs, one at a time,
and rolls the results into a single CSV.

```
sweep/grid.py          # the sweep plan: baseline + one-factor-at-a-time (OFAT) values
sweep/run_sweep.py      # driver: deploy -> bench -> pull, once per config, logged + resumable
sweep/aggregate.py      # scans results/*.json + the manifest -> results/aggregated.csv
sweep/manifest.jsonl    # append-only record of every config run (tag, params, status, timing)
sweep/logs/<tag>.log    # full run.sh output per config
config/generated/       # per-config ConfigMap YAMLs rendered by run_sweep.py
```

**Why OFAT, not full factorial:** sweeping N params with K values each is
K^N configs. `sweep/grid.py` instead holds everything at the baseline
(current `config/vllm-params.yaml` values) and changes **one** knob per run,
so cost is linear: `sum(len(values) - 1) + 1` configs. See it and its
estimated run count/time with:

```bash
python3 sweep/grid.py            # print the plan as a table
python3 sweep/run_sweep.py --dry-run   # plan + wall-clock estimate, no cluster calls
```

Currently swept (edit `SWEEP` in `sweep/grid.py` to add/remove/change values):

| Param | Values tried | ConfigMap key |
|---|---|---|
| `kv_cache_dtype` | auto, fp8, fp8_e4m3, fp8_e5m2 | `KV_CACHE_DTYPE` |
| `max_num_seqs` | 32, 64, 128, 256 | `MAX_NUM_SEQS` |
| `max_num_batched_tokens` | 512, 1024, 2048, 4096, 8192 | `MAX_NUM_BATCHED_TOKENS` |
| `attention_backend` | auto, FLASHINFER, TRITON_ATTN | `VLLM_ATTENTION_BACKEND` |
| `gpu_memory_utilization` | 0.80, 0.85, 0.90, 0.95 | `GPU_MEMORY_UTILIZATION` |
| `block_size` | 16, 32 | `BLOCK_SIZE` |
| `max_model_len` | 8192, 32768, 131072, 262144 | `MAX_MODEL_LEN` |

### Running the sweep

```bash
python3 sweep/run_sweep.py                      # full OFAT sweep, both benchmarks per config (~9h for 20 configs)
python3 sweep/run_sweep.py --bench rate         # faster: rate test only per config
python3 sweep/run_sweep.py --resume             # rerun after an interruption, skips completed configs
python3 sweep/run_sweep.py --only max-num-seqs  # rerun/debug just one parameter's variants
python3 sweep/run_sweep.py --stop-on-error      # abort on first failed config instead of continuing
```

Each config: renders `config/generated/vllm-params.<tag>.yaml` (baseline +
one override) → `PARAMS=<that file> ./run.sh all <bench> <tag>` (deploy +
wait Ready + benchmark) → `./run.sh pull <tag>` (always, even on failure, so
partial results aren't lost). Every attempt — success or failure — gets one
line in `sweep/manifest.jsonl` and a full log in `sweep/logs/<tag>.log`, so
the sweep can be safely killed and resumed. `run.sh` propagates a real
failing exit code when a benchmark Job fails on the cluster, so
`--resume` only skips tags that are genuinely done.

### Rerunning failed configs

`--resume` trusts `sweep/manifest.jsonl`, which only reflects reality for
runs made after the exit-code fix above. If you have older entries (or just
want to double check), audit the manifest against what's actually in
`results/` first:

```bash
python3 sweep/check_results.py                # report only: which tags are
                                                # missing/empty result files
python3 sweep/check_results.py --fix-manifest  # also append corrected
                                                # "failed" entries for any
                                                # stale "success" so...
python3 sweep/run_sweep.py --resume            # ...this only reruns those
```

`--resume` builds the full OFAT plan and drops any tag whose *last* manifest
entry is `"success"` — everything else (failed, interrupted, or never
attempted) gets rerun, deploy + both benchmarks, from scratch.

### Running a custom / one-off ConfigMap

To run a single config you authored yourself (hand-edited, or one of the
generated ones like `config/generated/vllm-params.recommended.yaml`)
through the same deploy → bench → pull → manifest pipeline as the sweep,
without touching the OFAT grid at all:

```bash
python3 sweep/run_sweep.py --custom-config config/generated/vllm-params.recommended.yaml --tag final-rerun
# or any ConfigMap yaml you wrote by hand, anywhere:
python3 sweep/run_sweep.py --custom-config /path/to/my-config.yaml --tag my-experiment
```

This records one entry in `sweep/manifest.jsonl` (`param: "custom"`) and one
log in `sweep/logs/<tag>.log`, just like a normal sweep point, so it shows
up alongside everything else if you rerun `sweep/aggregate.py`. `--only` and
`--resume` are ignored in this mode (there's no grid to filter). If you
don't care about manifest/log tracking, the lighter-weight equivalent is
just `PARAMS=/path/to/config.yaml ./run.sh all both <tag>` directly.

### Aggregating results

```bash
python3 sweep/aggregate.py
# -> results/aggregated.csv
```

One row per (config, test type, load level): `rate` results contribute 1 row,
`concurrent` results contribute up to 4 (one per stream level: 8/16/32/64).
Columns include the tag/param/value being swept, the full effective config
(`cfg_*`), request outcome counts + `error_rate`, and mean/p50/p90/p99 for
latency (`request_latency_s`, `ttft_ms`, `tpot_ms`, `itl_ms`) and throughput
(`requests_per_second`, `output_tokens_per_second`, `tokens_per_second`,
`prompt_tokens_per_second`).

### Plotting + picking a final config

```bash
pip install matplotlib pandas pyyaml   # once
python3 sweep/plot_results.py
```

Reads `results/aggregated.csv` and writes PNGs to `results/plots/`:

- `00_overview_rate.png` / `00_overview_concurrent.png` — every tag/run
  compared head-to-head on TTFT p50/p99, latency p50/p99, throughput, error
  rate.
- `01_latency_throughput_frontier.png` — latency-vs-throughput tradeoff
  curve for every config (p50 and p99 side by side), annotated by
  concurrency level.
- `param_<name>.png` — one per swept parameter: bar chart at the rate test
  + line chart of how it scales across concurrency.
- `param_rankings.csv` — the leaderboard table printed to stdout.

It also ranks each parameter's tested values (excluding any whose error rate
is too high to trust — see the script's `ERROR_RATE_THRESHOLD`, since a
config that errors out most requests looks artificially *fast* on the
survivors) and writes the best independently-tested value per parameter to
`config/generated/vllm-params.recommended.yaml`, ready for a confirmation
rerun:

```bash
PARAMS=config/generated/vllm-params.recommended.yaml ./run.sh all both final-rerun
```

Because the sweep is OFAT, this combines winners found independently, not a
jointly-optimized combo — treat the rerun as a confirmation step. The
script is split into `# %%` cells, so it also opens as a notebook in
Cursor/VS Code without installing Jupyter.

### Comparing two specific tags head-to-head

`plot_results.py` renders every tag at once, which is noisy when you just
want config A vs. config B (e.g. baseline vs. a candidate, or two finalists):

```bash
python3 sweep/compare_tags.py baseline max-num-seqs-256
# fuzzy tag matching (unique substring) + custom legend labels also work:
python3 sweep/compare_tags.py baseline final-rerun --label-a "current prod" --label-b "recommended"
```

Prints a delta table (tag_b vs. tag_a, % change) to stdout and writes to
`results/plots/compare/`:

- `<tag_a>_vs_<tag_b>_rate.png` — grouped bar chart at the rate test for
  TTFT p50/p99, ITL p50/p99, end-to-end latency p50/p99, mean throughput,
  and error rate.
- `<tag_a>_vs_<tag_b>_concurrent.png` — same metrics as line charts scaling
  across the 8/16/32/64 concurrency levels.
- `<tag_a>_vs_<tag_b>_frontier.png` — latency-vs-throughput frontier for
  just these two tags.

## SGLang (later)

Adding SGLang is symmetric: a second Deployment (`strategy: Recreate`, same GPU
+ model-cache PVC) fronted by its own Service, plus `--model-path` /
`--chunked-prefill-size` / `--max-running-requests` flags instead of the vLLM
ones. Because of the single-GPU rule you'll deploy one engine at a time; the
benchmark Jobs just point at whichever Service is live.
