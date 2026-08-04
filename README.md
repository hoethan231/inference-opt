# Inference optimization bench

One tool — `./bench` — for measuring vLLM serving configurations on the cluster.
Single runs, one-factor-at-a-time sweeps and grid searches are the same thing at
different sizes, so they share one command, one results store and one report.

# Inference optimization bench

One tool — `./bench` — for measuring vLLM serving configurations on the cluster.
Single runs, one-factor-at-a-time sweeps and grid searches are the same thing at
different sizes, so they share one command, one results store and one report.

```bash
export HF_TOKEN=hf_xxx              # first time only; creates the Secret

./bench models                      # what can be run: models, workloads, experiments
./bench run baseline                # deploy one config, benchmark it, record it
./bench report baseline             # plots
```

Everything else is a variation on that. `./bench` with no arguments lists the
commands; `./bench <command> --help` explains one.

## How a run works

A run is the same four steps the old `run.sh` did, in a loop, per config:

1. **Render** the `vllm-params` ConfigMap for this config and apply it.
2. **Refresh** the server — the Deployment is `strategy: Recreate`, so the old
  pod releases the GPU before the new one starts — then wait until it is Ready
   and answering, and record what the engine actually did (KV-cache blocks,
   resolved engine args, vLLM version).
3. **Benchmark** it with each workload, as a Kubernetes Job.
4. **Pull** the results off the PVC into `runs/<tag>/` and append one line to
  `runs/index.jsonl`.

Nothing about that loop is hidden: `./bench render <experiment>` prints the exact
manifests a run would apply, and applies nothing.

### What gets recorded, and where to read it

`./bench show <tag>` prints everything the store knows about one run, and is the
answer to "what was this config actually served with?":

* **Config as applied** — every key of the ConfigMap, with the ones this run set
  marked, and `EXTRA_ARGS` split one flag per line. Copied from the manifest that
  was applied, so it cannot drift from what ran. The same thing is on disk as
  `runs/<tag>/config.yaml`.
* **Engine facts** — vLLM version, KV cache size in tokens and GiB, the maximum
  concurrency vLLM computed for that cache, and graph capture time, all read out
  of the server's own startup log (`runs/<tag>/server.log`) once it goes Ready. A
  config left with only a sliver of KV cache is slow for a reason no latency chart
  shows. Alongside them, from the labels of the node the pod landed on: the GPU
  model, how many that node has, the driver version and the node name — so moving
  host reads as an environment change rather than a regression in whatever was
  being swept.
* **Outcome** — status, how long it took, the smoke-test response, and for a
  failure the evidence line that caused it to be abandoned.
* **Files** — the raw GuideLLM JSON and HTML per workload, the benchmark job logs,
  and `engine.log`, which is the server log captured *during* the benchmark rather
  than at startup.

Runs imported by `bench migrate` have a config but no engine facts: the old
scripts never captured them.

## Defining what to measure

An experiment is one YAML file under `experiments/`, filed by how many GPUs it
needs (`single-gpu/`, `multi-gpu/`). Its `strategy.kind` decides how many configs
come out of it:


| kind     | configs                    | use it for                                            |
| -------- | -------------------------- | ----------------------------------------------------- |
| `single` | 1                          | one config, e.g. a baseline or a candidate to confirm |
| `ofat`   | `sum(len(values) - 1) + 1` | finding each knob's best value on its own             |
| `grid`   | product of the axes        | proving a *combination*, once OFAT has narrowed it    |


An axis usually names a ConfigMap key:

```yaml
name: ofat-scheduler
model: nemotron-3-super-120b
workloads: [rate, concurrent]

strategy:
  kind: ofat
  axes:
    - key: MAX_NUM_SEQS
      note: max concurrent running sequences — the scheduler's bouncer
      values: ["32", "64", "128", "256"]
```

...but a value can change several things at once, or append a flag instead of
setting a key — which is how speculative decoding, kernel probes and anything
else without its own ConfigMap key are expressed:

```yaml
    - param: spec_tokens
      abbrev: mtp
      extra_args: >-
        --speculative-config={{"method":"{spec_decode_method}","num_speculative_tokens":{value}}}
      values: ["3", "4", "5"]

    - param: feature
      values:
        - label: cutlass-moe
          params: {MAX_NUM_SEQS: "256"}
          extra_args: --moe-backend=flashinfer_cutlass
```

`{spec_decode_method}` and `{hf_repo}` come from the model registry, so pointing
the experiment at another model picks up that model's method rather than
hardcoding one. `abbrev:` fixes how the axis appears in tags
(`spec-decode-mtp-5`); without it a tag spells the axis out and only falls back
to an abbreviation when it would otherwise exceed the Kubernetes name limit.
Each shipped experiment's header comment explains what it is for.

An experiment is named by its file, and the group is optional when you type it:
`./bench run baseline` and `./bench run single-gpu/baseline` are the same thing,
and the group only has to be spelled out if two groups hold the same name.

### Where a config's values come from

Lowest precedence to highest:

```
models/<model>.yaml  <  experiments/**/<exp>.yaml  <  the axis value
```

Model identity (`MODEL_NAME`, `SERVED_MODEL_NAME`) is applied last and can't be
set by an experiment, so an experiment can never quietly serve a different
checkpoint than the one in its `model:` field.

## Multiple models

A model file owns everything model-specific — weights, cache volume, image, the
parser flags its chat template needs, and defaults for knobs whose ceiling is
model-dependent:

```yaml
name: nemotron-3-super-120b
hf_repo: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
cache_pvc: nemotron-model-cache
params:
  MAX_MODEL_LEN: "262144"
extra_args: [--trust-remote-code, --tool-call-parser=qwen3_coder]
spec_decode_method: nemotron_h_mtp
```

Switching an experiment to another model is one line (`model: laguna-s-2.1`). If
its weights aren't on the cache volume yet, `bench run` notices and pre-seeds the
download first (a Job on whichever node has the GPU — no hostname is hardcoded).
If the model cache PVC does not exist yet, preflight creates it automatically
from `cluster/deployments/model-cache-pvc.yaml`.
Reports label each series with its model when more than one is in view, so
cross-model comparison happens at report time rather than needing its own mode.

## Running

| `single` | 1 | one config, e.g. a baseline or a candidate to confirm |
| `ofat` | `sum(len(values) - 1) + 1` | finding each knob's best value on its own |
| `grid` | product of the axes | proving a *combination*, once OFAT has narrowed it |

An axis usually names a ConfigMap key:

```yaml
name: ofat-scheduler
model: nemotron-3-super-120b
workloads: [rate, concurrent]

strategy:
  kind: ofat
  axes:
    - key: MAX_NUM_SEQS
      note: max concurrent running sequences — the scheduler's bouncer
      values: ["32", "64", "128", "256"]
```

...but a value can change several things at once, or append a flag instead of
setting a key — which is how speculative decoding, kernel probes and anything
else without its own ConfigMap key are expressed:

```yaml
    - param: spec_tokens
      abbrev: mtp
      extra_args: >-
        --speculative-config={{"method":"{spec_decode_method}","num_speculative_tokens":{value}}}
      values: ["3", "4", "5"]

    - param: feature
      values:
        - label: cutlass-moe
          params: {MAX_NUM_SEQS: "256"}
          extra_args: --moe-backend=flashinfer_cutlass
```

`{spec_decode_method}` and `{hf_repo}` come from the model registry, so pointing
the experiment at another model picks up that model's method rather than
hardcoding one. `abbrev:` fixes how the axis appears in tags
(`spec-decode-mtp-5`); without it a tag spells the axis out and only falls back
to an abbreviation when it would otherwise exceed the Kubernetes name limit.
The shipped experiments are `baseline`, `ofat-scheduler`,
`grid-scheduler`, `spec-decode`, `feature-probes` and `laguna-grid`; each one's
header comment explains what it is for.

### Where a config's values come from

Lowest precedence to highest:

```
cluster/params-base.yaml  <  models/<model>.yaml  <  experiments/<exp>.yaml  <  the axis value
```

Model identity (`MODEL_NAME`, `SERVED_MODEL_NAME`) is applied last and can't be
set by an experiment, so an experiment can never quietly serve a different
checkpoint than the one in its `model:` field.

## Multiple models

A model file owns everything model-specific — weights, cache volume, image, the
parser flags its chat template needs, and defaults for knobs whose ceiling is
model-dependent:

```yaml
name: nemotron-3-super-120b
hf_repo: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
cache_pvc: nemotron-model-cache
params:
  MAX_MODEL_LEN: "262144"
extra_args: [--trust-remote-code, --tool-call-parser=qwen3_coder]
spec_decode_method: nemotron_h_mtp
```

Switching an experiment to another model is one line (`model: laguna-s-2.1`). If
its weights aren't on the cache volume yet, `bench run` notices and pre-seeds the
download first (a Job on whichever node has the GPU — no hostname is hardcoded).
Reports label each series with its model when more than one is in view, so
cross-model comparison happens at report time rather than needing its own mode.

## Running

```bash
./bench run ofat-scheduler --dry-run          # the plan and the wall-clock estimate
./bench run ofat-scheduler                    # all of it
./bench run ofat-scheduler --only max_num_seqs  # one axis (matches tag, axis or value)
./bench run ofat-scheduler --resume           # carry on, skipping what already succeeded
./bench run baseline --workloads rate         # cheaper: one workload
./bench run ofat-scheduler --suffix rerun2    # same sweep, tags suffixed, nothing overwritten
./bench run external/example
```

Safe to interrupt: every attempt is recorded, so `--resume` picks up exactly
what is left. A config that fails is retried once with a fresh deploy — except
when it failed because it cannot fit in memory, which retrying cannot fix.

### Sweeping externally hosted endpoints

When the model is served outside this cluster (for example an enterprise-hosted
OpenAI-compatible endpoint), define an external manifest under
`experiments/external/`. Anything in that directory runs in external mode
automatically (no `--external` flag needed).

Example (one file per run), `experiments/external/mns-64-mnbt-1024.yaml`:

```yaml
name: external
model: nemotron-3-super-120b
request_model: nemotr-super-nvfp4
endpoint: https://host/v1/chat/completions
apikey: sk-...
workloads: [rate, concurrent]
runs:
  - mns-64-mnbt-1024
## Reading the results

./bench run ofat-scheduler --dry-run          # the plan and the wall-clock estimate
./bench run ofat-scheduler                    # all of it
./bench run ofat-scheduler --only max_num_seqs  # one axis (matches tag, axis or value)
./bench run ofat-scheduler --resume           # carry on, skipping what already succeeded
./bench run baseline --workloads rate         # cheaper: one workload
./bench run ofat-scheduler --suffix rerun2    # same sweep, tags suffixed, nothing overwritten

```

Run it:

```bash
./bench run external/mns-64-mnbt-1024
```

Repeat with one manifest per config after you manually restart/reconfigure the
external server for the next point.

By convention, external names are `param-val-param-val` (for example
`mns-64-mnbt-1024`). Reports parse these tags into per-axis views automatically.

External runs are stored under `runs/<tag>/` like all other runs; their tags
start with `external-` so they are easy to distinguish.

`model` must be one of this repo's local model registry files under `models/`.
For externally hosted gateways where the request `"model"` differs, set
`request_model` (or `served_model`) to the remote model ID.

External experiments use a dedicated benchmark Job template
(`cluster/jobs/benchmark-external.yaml`) and inject your manifest's endpoint
and `request_model` into the scenario at runtime. This lets multiple external
manifests target different endpoints in parallel.

External benchmark jobs also write to a separate PVC (`external_results_pvc` in
`cluster/cluster.yaml`) so they do not contend with local in-cluster sweeps on
ReadWriteOnce storage.

Manifest endpoint is reused for all runs in that file. You can still override
the API key at runtime:

```bash
./bench run external/mns-64-mnbt-1024 --external-api-key "$OPENAI_API_KEY"
```

To plot all external runs together (overview/frontier + per-axis sweep detail):

```bash
./bench report external
```

### Running local and external at once

Yes. You can run a normal in-cluster sweep and an external manifest run in
parallel (separate terminals). Benchmark Jobs no longer read model identity
from the shared params ConfigMap, which avoids cross-run tokenizer/model races.

`--suffix rerun2` turns `max-num-seqs-256` into `max-num-seqs-256-rerun2`.
Reports treat a suffix as a separate group and keep it out of the default view,
because it is usually a different question (a different workload, or a rerun);
`--suffix rerun2` plots that group, `--all-suffixes` plots everything together.

## Reading the results

```bash
./bench runs                          # what has been measured, with tags
./bench show <tag>                    # one run in full: config, engine, files
./bench report ofat-scheduler         # overview, frontier and per-axis figures
./bench report ofat-scheduler --param max_num_seqs   # drill into one axis
./bench report --all                  # everything, every model, on shared axes
./bench compare baseline spec-decode-mtp-5           # head-to-head, with numbers
```

PNGs land in `reports/<experiment>/`. There are no CSVs and no aggregation step —
the report reads the run store directly. Parsed metrics are cached per run
(`runs/<tag>/summary.json`, fingerprinted by size and mtime), which is the
difference between a report taking milliseconds and re-parsing gigabytes of
GuideLLM JSON.

Each figure pairs p50 with p99: TTFT, inter-token latency and output throughput.
`bench report` draws every run in view; `bench compare` is the one that prints
numbers, as a delta table between exactly two runs.

**There is no ranking, no score and no leaderboard.** Which config is "best"
depends on whether you are optimising for time-to-first-token, tail latency or
tokens per second, and collapsing those into one number hides the tradeoff the
plots exist to show. Pick from the frontier plot, then confirm with `compare`.

Two things a plot cannot show, so the report says them in words: a run that
completed no requests at all (its lines sit at zero, which looks fast), and a run
whose error rate is above 5% (its latencies describe only the requests that
survived). Both are listed after the figures, and `bench doctor` finds them too.

A series label names whatever varies besides the axis — its model, and its GPU
topology — so a two-GPU run is never drawn as though it were a two-GPU-shaped
result for a one-GPU config. `bench compare` says so explicitly when the two runs
it was given did not use the same number of GPUs.

### About throughput

Throughput in the plots is GuideLLM's `output_tokens_per_second`, not
`tokens_per_second`. The latter is not usable as a distribution: it reports a
whole-run aggregate as its mean but windowed samples as its percentiles, so one
run can show a mean of 15,064 tok/s, a p50 of 36 and a p99 of 405,869. Output
tokens per second is internally consistent, so its p50 and p99 mean what they say.

Note that output throughput therefore excludes prompt tokens. On this
coding-agent traffic, with its large shared prefixes, total token rate is roughly
two orders of magnitude higher than the output rate.

GuideLLM's own HTML report is pulled alongside the JSON in `runs/<tag>/` when you
want the per-request detail behind a number.

## Keeping the store honest

```bash
./bench doctor            # index vs. what's actually on disk
./bench doctor --fix      # mark unusable runs failed, so --resume redoes them
```

`doctor` catches a run recorded as successful whose result file is missing,
truncated, or contains zero successful requests. `runs/index.jsonl` is
append-only — a record is never rewritten, so retries and a killed sweep's
history both stay visible, and readers take the last entry per tag.

## Cluster commands

`doctor` catches a run recorded as successful whose result file is missing,
truncated, or contains zero successful requests. `runs/index.jsonl` is
append-only — a record is never rewritten, so retries and a killed sweep's
history both stay visible, and readers take the last entry per tag.

## Cluster commands

```bash
./bench status            # server, GPUs, benchmark jobs, recent runs
./bench pull [tag]        # raw files off the results PVC
./bench clean jobs        # delete finished benchmark Jobs
./bench clean results     # wipe the results PVC (asks first; pull anything you want)
./bench models --preseed laguna-s-2.1   # download weights to the cache volume
```

## Layout

```
bench                     # the entry point (a shim around `uv run`)
benchlab/                 # the tool
experiments/
  single-gpu/*.yaml       # what to measure on one GPU
  multi-gpu/*.yaml        # what to measure across both
models/*.yaml             # which model, and what it needs
workloads/                # GuideLLM scenarios + index.yaml
cluster/
  cluster.yaml            # every Kubernetes object name, in one file
  deployments/            # vLLM Deployment + Service, results PVC
  jobs/                   # benchmark Job, weight pre-seed Job
runs/
  index.jsonl             # append-only record of every attempt
  <tag>/                  # result JSON + HTML, engine facts, logs
reports/                  # PNGs, one directory per experiment
archive/                  # the pre-refactor tools, configs and outputs
```

`cluster/cluster.yaml` holds every Kubernetes name the tool uses, deliberately
in one place so none of them is buried in Python. The manifests under `cluster/`
are the same ones as before, with `__PLACEHOLDER__` fields the tool fills in per
run.

## The 30-second Kubernetes primer

Everything lives in the `vllm-inf` **namespace**. You never talk to a machine
directly; you declare **objects** and Kubernetes makes reality match them.


| Object         | What it is                              | Ours                                                                 |
| -------------- | --------------------------------------- | -------------------------------------------------------------------- |
| **ConfigMap**  | key/value settings injected as env vars | `vllm-params` (the knobs), `bench-workloads` (scenarios)             |
| **Secret**     | a ConfigMap for sensitive values        | `hf-token-secret`                                                    |
| **PVC**        | a network disk that outlives pods       | `nemotron-model-cache` (weights), `vllm-sweep-results-pvc` (results) |
| **Deployment** | keeps N copies of a pod running         | `vllm-server`                                                        |
| **Service**    | a stable DNS name in front of pods      | `vllm-svc`                                                           |
| **Job**        | a pod that runs once to completion      | each benchmark, each pre-seed                                        |


A pod is isolated: it only sees settings you inject and disks you mount. That is
why the token is a Secret and the weights are on a PVC.

**One engine at a time, on two GPUs.** The node has two GPUs, and a config may
use both (see below), so `bench run` works out how many devices the config needs
and only scales down the rival engines in `cluster.yaml` when it is actually short
— a colleague's single-GPU job can coexist with a single-GPU run of ours. If
something else still holds the devices it waits (`--gpu-wait-timeout`) rather
than deadlocking against it.

What stops two of *our* engines running side by side is host memory, not GPUs:
the pod requests 40Gi and the GPU node has ~47Gi allocatable. That is deliberate
rather than a limitation to work around — two engines on one node would share
PCIe, host memory bandwidth and the weights volume, and neither one's latency
numbers would be comparable to anything else in the store. The second GPU is
capacity for one engine, not a second lane for the sweep.

**Why the results PVC is separate.** `vllm-sweep-results-pvc` belongs to this
repo, so it can't fill up from — or be cleared alongside — other benchmarking
work on the shared cluster. Never point these Jobs at the shared
`guidellm-results-pvc`.

## The knobs

The Deployment turns ConfigMap keys into vLLM flags, so re-tuning never means
editing the Deployment:

| Key | vLLM flag |
|---|---|
| `MAX_MODEL_LEN` | `--max-model-len` — context window |
| `MAX_NUM_SEQS` | `--max-num-seqs` — max concurrent running requests |
| `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` — prefill chunk size |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` — `fp8` roughly halves KV-cache size |
| `GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` |
| `TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` |
| `PIPELINE_PARALLEL_SIZE` | `--pipeline-parallel-size` |
| `DATA_PARALLEL_SIZE` | `--data-parallel-size` |
| `EXPERT_PARALLEL` | `1` appends `--enable-expert-parallel` (MoE, more than one rank) |
| `BLOCK_SIZE` | `--block-size` |
| `VLLM_ATTENTION_BACKEND` | read as an env var; empty means vLLM auto-selects |
| `EXTRA_ARGS` | appended verbatim — the escape hatch for everything else |

`EXTRA_ARGS` is composed from the `extra_args` lists (model, then
experiment, then axis) and never set directly, which is how configs used to drift
apart. The entrypoint splits it with `eval set --`, so a flag whose value
contains spaces or JSON survives.

### Spreading one engine over both GPUs

The three size knobs multiply into the number of GPUs the pod requests —
`TENSOR × PIPELINE × DATA` — and that arithmetic, not a hand-written number, is
what fills in the Deployment's `nvidia.com/gpu`
(`benchlab/parallelism.py`). A config asking for more devices than a node has is
refused when the plan is built, because the failure it would otherwise cause is
slow and misleading: a pod stuck Pending, or an engine hanging on a rank that
cannot start, recorded as a failure of whatever knob the sweep was measuring.

The four are not interchangeable:

| Knob | What it does | Costs |
|---|---|---|
| tensor | shards every layer across ranks | an all-reduce per forward pass |
| pipeline | gives each rank a slice of the layers | one handoff per stage, and bubbles |
| data | whole replicas behind one endpoint | nothing shared, so no comms — but no extra KV per replica either |
| expert | spreads an MoE's experts over the ranks | needs more than one rank |

On this hardware the trade is sharp, because the two RTX PRO 6000 Blackwell cards
have **no NVLink** — tensor-parallel traffic crosses PCIe Gen5. Against that,
tensor parallelism is what buys KV cache: at `TENSOR_PARALLEL_SIZE 1` the 120B
NVFP4 weights take 69.62 GiB of the 85.47 GiB budget and the cache gets the
remaining 13.61 GiB, which vLLM reports as 2.23x maximum concurrency at 262k
context. `experiments/multi-gpu/tp-scaling.yaml` measures that trade rather than
assuming it.

## The workloads

All of them hit `/v1/chat/completions` with synthetic coding-agent traffic (large
shared prefixes, bursty prompt/output sizes, tool-heavy conversations). They
differ in how load is applied — a workload's shape is read from the scenario's
own `spec.profile.kind`, so adding one is just a `workloads/index.yaml` entry.


| Name           | Load                                                            | Time    |
| -------------- | --------------------------------------------------------------- | ------- |
| `rate`         | steady 20 req/s Poisson stream — behavior at fixed offered load | ~6 min  |
| `concurrent`   | 8/16/32/64 busy users — how latency and throughput scale        | ~20 min |
| `decode-heavy` | short prompts, long generations — isolates decode from prefill  | ~6 min  |


## Requirements

`./bench` is a shim: it runs the tool through [uv](https://docs.astral.sh/uv/),
which fetches Python 3.12 and the pinned dependencies on first use. You need
`uv` and a working `kubectl` context. Nothing else, and no virtualenv to
activate.

On the cluster side: one GPU node (2x RTX PRO 6000 Blackwell Server Edition,
~96 GB each, PCIe Gen5, no NVLink), the default `nutanix-volume` StorageClass,
and a model cache PVC. `HF_TOKEN` in the environment on first run creates the
Secret; the results PVC is created if it is missing. `gpus_per_node` in
`cluster/cluster.yaml` is the declared device count — declared rather than
discovered so `--dry-run` and `bench render` need no cluster access — so update it
if the node changes.

## What happened to the old scripts

`run.sh`, `sweep/run_sweep.py`, `sweep/run_grid.py`, `sweep/plot_results.py`,
`sweep/plot_grid.py`, `sweep/compare_tags.py`, `sweep/load_results.py`,
`sweep/check_results.py` and `sweep/grid.py` are all folded into `./bench` and
parked in `archive/legacy-tools/` for reference. The old manifests and result
files were imported (`./bench migrate`, already run), so pre-refactor runs are in
the store under `legacy-*` experiment names and still comparable:

```bash
./bench compare gpu-memory-utilization-0-85 gpu-memory-utilization-0-95
```


| Then                                                                    | Now                                           |
| ----------------------------------------------------------------------- | --------------------------------------------- |
| edit `config/vllm-params.yaml`, `./run.sh all both`                     | `./bench run <experiment>`                    |
| `sweep/grid.py` `SWEEP` list                                            | `experiments/*.yaml`                          |
| `config/grid.json` + `run_grid.py`                                      | `strategy.kind: grid`                         |
| `config/custom/*.yaml` one-offs                                         | one `ofat` axis of presets (`feature-probes`) |
| two manifests + `param: "grid"`                                         | `runs/index.jsonl`                            |
| `plot_results.py` / `plot_grid.py` + CSVs                               | `./bench report`                              |
| `check_results.py --fix-manifest`                                       | `./bench doctor --fix`                        |
| ./bench compare gpu-memory-utilization-0-85 gpu-memory-utilization-0-95 |                                               |
| ```                                                                     |                                               |





| Then                                                | Now                                           |
| --------------------------------------------------- | --------------------------------------------- |
| edit `config/vllm-params.yaml`, `./run.sh all both` | `./bench run <experiment>`                    |
| `sweep/grid.py` `SWEEP` list                        | `experiments/*.yaml`                          |
| `config/grid.json` + `run_grid.py`                  | `strategy.kind: grid`                         |
| `config/custom/*.yaml` one-offs                     | one `ofat` axis of presets (`feature-probes`) |
| two manifests + `param: "grid"`                     | `runs/index.jsonl`                            |
| `plot_results.py` / `plot_grid.py` + CSVs           | `./bench report`                              |
| `check_results.py --fix-manifest`                   | `./bench doctor --fix`                        |




## SGLang (later)

Symmetric to vLLM: a second Deployment (`strategy: Recreate`, same GPUs and cache
PVC) behind its own Service, with `--model-path` / `--chunked-prefill-size` /
`--max-running-requests` instead of the vLLM flags. Because of the single-GPU
rule you would still run one engine at a time; the benchmark Jobs point at
whichever Service is live.
Symmetric to vLLM: a second Deployment (`strategy: Recreate`, same GPU and cache
PVC) behind its own Service, with `--model-path` / `--chunked-prefill-size` /
`--max-running-requests` instead of the vLLM flags. Because of the single-GPU
rule you would still run one engine at a time; the benchmark Jobs point at
whichever Service is live.
