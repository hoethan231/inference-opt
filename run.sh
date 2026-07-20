#!/usr/bin/env bash
# =============================================================================
# run.sh  ·  Thin driver for the vLLM benchmarking loop.
#
#   ./run.sh deploy              apply params + (re)start the server, wait ready
#   ./run.sh bench [test] [tag]  run a benchmark Job   (test: rate|concurrent|both)
#   ./run.sh all   [test] [tag]  deploy, then bench
#   ./run.sh pull  [tag]         copy result files off the PVC to ./results/
#   ./run.sh status              show server + benchmark Jobs
#   ./run.sh clean               delete this repo's benchmark Jobs
#   ./run.sh clean-results [tag] delete result files off the PVC (pull first!)
#
# The whole tuning loop is:
#   1. edit config/vllm-params.yaml
#   2. ./run.sh deploy
#   3. ./run.sh bench both
#   4. ./run.sh pull
#
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=vllm-inf
DEPLOY="$HERE/deployments/vllm.yaml"
RESULTS_PVC_YAML="$HERE/deployments/results-pvc.yaml"
RESULTS_PVC="vllm-sweep-results-pvc"   # this repo's own volume; see deployments/results-pvc.yaml
PARAMS="${PARAMS:-$HERE/config/vllm-params.yaml}"
SCEN_DIR="$HERE/scenarios"
JOBS="$HERE/jobs"
RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx \033[0m %s\n' "$*" >&2; exit 1; }

command -v kubectl >/dev/null || die "kubectl not found on PATH"

# Sanitize a string into an RFC-1123 label (lowercase alnum + dashes).
sanitize() { echo "$1" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9-' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//'; }

# Ensure the HF token Secret exists (recreate it if HF_TOKEN is exported).
ensure_secret() {
  if [ -n "${HF_TOKEN:-}" ]; then
    log "creating/updating Secret hf-token-secret from \$HF_TOKEN"
    kubectl create secret generic hf-token-secret -n "$NS" \
      --from-literal=HF_TOKEN="$HF_TOKEN" \
      --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  elif ! kubectl get secret hf-token-secret -n "$NS" >/dev/null 2>&1; then
    die "Secret hf-token-secret missing and \$HF_TOKEN not set. Run: export HF_TOKEN=hf_xxx"
  fi
}

# Apply the params ConfigMap and (re)build the scenarios ConfigMap from files.
apply_config() {
  log "applying params ConfigMap (config/vllm-params.yaml)"
  kubectl apply -f "$PARAMS" >/dev/null
  log "applying scenarios ConfigMap (from scenarios/*.json)"
  kubectl create configmap coding-agent-scenarios -n "$NS" \
    --from-file="$SCEN_DIR/coding-agent-rate.json" \
    --from-file="$SCEN_DIR/coding-agent-users.json" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
}

# Free the single GPU: scale down any other engine that might be holding it.
free_gpu() {
  log "freeing GPU (scaling down any other engine holding it)"
  kubectl scale deployment sglang-nemotron -n "$NS" --replicas=0 >/dev/null 2>&1 || true
  kubectl scale statefulset vllm-nemotron  -n "$NS" --replicas=0 >/dev/null 2>&1 || true
  kubectl wait --for=delete pod -l app=sglang-nemotron -n "$NS" --timeout=180s >/dev/null 2>&1 || true
  kubectl wait --for=delete pod -l app=vllm-nemotron  -n "$NS" --timeout=180s >/dev/null 2>&1 || true
}

cmd_deploy() {
  ensure_secret
  apply_config
  log "ensuring our own results PVC exists ($RESULTS_PVC)"
  kubectl apply -f "$RESULTS_PVC_YAML" >/dev/null
  free_gpu
  log "applying Deployment + Service"
  kubectl apply -f "$DEPLOY" >/dev/null
  # Force a fresh pod so ConfigMap changes are always picked up, then wait.
  log "restarting server to pick up new params"
  kubectl rollout restart deployment/vllm-server -n "$NS" >/dev/null
  log "waiting for vllm-server to become Ready (model load can take minutes)..."
  kubectl rollout status deployment/vllm-server -n "$NS" --timeout=60m
  log "vllm-server is READY."
}

# Run a single benchmark Job from a template, waiting for it to finish.
run_one() {
  local test="$1" tag="$2" tmpl name
  case "$test" in
    rate)       tmpl="$JOBS/vllm_rate.yaml"; name="vllm-rate-$tag" ;;
    concurrent) tmpl="$JOBS/vllm_user.yaml"; name="vllm-users-$tag" ;;
    *) die "unknown test '$test' (expected rate|concurrent)" ;;
  esac
  log "running '$test' benchmark  (job: $name)"
  kubectl delete job "$name" -n "$NS" --ignore-not-found >/dev/null 2>&1 || true
  sed "s/__TAG__/$tag/g" "$tmpl" | kubectl apply -f - >/dev/null
  # Stream logs once the pod starts, then block on completion.
  kubectl wait --for=condition=Ready pod -l job-name="$name" -n "$NS" --timeout=300s >/dev/null 2>&1 || true
  kubectl logs -f job/"$name" -n "$NS" 2>/dev/null || true
  if kubectl wait --for=condition=complete job/"$name" -n "$NS" --timeout=5s >/dev/null 2>&1; then
    log "  $test -> COMPLETE  (results file: ${name}.json)"
  elif kubectl get job "$name" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null | grep -q 1; then
    warn "  $test -> FAILED. Inspect: kubectl logs job/$name -n $NS"
  else
    warn "  $test -> still running/unknown. Check: kubectl get job $name -n $NS"
  fi
}

cmd_bench() {
  local test="${1:-both}" tag
  tag="$(sanitize "${2:-${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}}")"
  apply_config   # make sure scenarios exist / are current
  log "benchmark tag = $tag"
  case "$test" in
    rate)       run_one rate "$tag" ;;
    concurrent) run_one concurrent "$tag" ;;
    both)       run_one rate "$tag"; run_one concurrent "$tag" ;;
    *) die "unknown test '$test' (expected rate|concurrent|both)" ;;
  esac
  log "pull results with: ./run.sh pull $tag"
}

cmd_all() { cmd_deploy; cmd_bench "${1:-both}" "${2:-}"; }

# Copy result files off the results PVC using an ephemeral reader pod.
cmd_pull() {
  local tag="${1:-}" glob="vllm-*"
  [ -n "$tag" ] && glob="vllm-*${tag}*"
  mkdir -p "$RESULTS_DIR"
  log "pulling results ($glob*) -> $RESULTS_DIR"
  kubectl delete pod bench-reader -n "$NS" --ignore-not-found --wait=true >/dev/null 2>&1 || true
  cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata: { name: bench-reader, namespace: ${NS}, labels: { app: bench-reader } }
spec:
  restartPolicy: Never
  containers:
    - name: reader
      image: busybox:1.36
      command: ["sleep","3600"]
      volumeMounts: [ { name: results, mountPath: /results, readOnly: true } ]
  volumes:
    - { name: results, persistentVolumeClaim: { claimName: ${RESULTS_PVC} } }
YAML
  kubectl wait --for=condition=Ready pod/bench-reader -n "$NS" --timeout=120s >/dev/null
  local f
  for f in $(kubectl exec bench-reader -n "$NS" -- sh -c "ls /results/${glob} 2>/dev/null"); do
    log "  $(basename "$f")"
    kubectl exec bench-reader -n "$NS" -- cat "$f" > "$RESULTS_DIR/$(basename "$f")"
  done
  kubectl delete pod bench-reader -n "$NS" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  log "done -> $RESULTS_DIR"
}

# Delete result files off the results PVC using an ephemeral writer pod.
# ALWAYS pull first — this is permanent. Only touches this repo's own PVC
# ($RESULTS_PVC), never the shared guidellm-results-pvc other tools use.
cmd_clean_results() {
  local glob="vllm-*"
  [ -n "${1:-}" ] && glob="vllm-*${1}*"
  warn "deleting '${glob}' from PVC '${RESULTS_PVC}' -- make sure you've run './run.sh pull' first!"
  read -r -p "Type 'yes' to continue: " confirm
  [ "$confirm" = "yes" ] || die "aborted"
  kubectl delete pod bench-cleaner -n "$NS" --ignore-not-found --wait=true >/dev/null 2>&1 || true
  cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata: { name: bench-cleaner, namespace: ${NS}, labels: { app: bench-cleaner } }
spec:
  restartPolicy: Never
  containers:
    - name: cleaner
      image: busybox:1.36
      command: ["sleep","300"]
      volumeMounts: [ { name: results, mountPath: /results } ]
  volumes:
    - { name: results, persistentVolumeClaim: { claimName: ${RESULTS_PVC} } }
YAML
  kubectl wait --for=condition=Ready pod/bench-cleaner -n "$NS" --timeout=120s >/dev/null
  kubectl exec bench-cleaner -n "$NS" -- sh -c "rm -fv /results/${glob}"
  kubectl delete pod bench-cleaner -n "$NS" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  log "done"
}

cmd_status() {
  log "server"
  kubectl get deployment vllm-server -n "$NS" \
    -o custom-columns='NAME:.metadata.name,READY:.status.readyReplicas,DESIRED:.spec.replicas' 2>/dev/null || true
  kubectl get pods -n "$NS" -l app=vllm -o wide 2>/dev/null || true
  echo
  log "benchmark jobs (suite=vllm)"
  kubectl get jobs -n "$NS" -l bench.suite=vllm \
    -o custom-columns='NAME:.metadata.name,SUCCEEDED:.status.succeeded,FAILED:.status.failed,AGE:.metadata.creationTimestamp' 2>/dev/null || true
}

cmd_clean() {
  log "deleting benchmark jobs (suite=vllm)"
  kubectl delete job -n "$NS" -l bench.suite=vllm --ignore-not-found
}

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

main() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    deploy)        cmd_deploy ;;
    bench)         cmd_bench "$@" ;;
    all)           cmd_all "$@" ;;
    pull)          cmd_pull "$@" ;;
    status)        cmd_status ;;
    clean)         cmd_clean ;;
    clean-results) cmd_clean_results "$@" ;;
    help|-h|--help) usage ;;
    *) usage; die "unknown command: $cmd" ;;
  esac
}
main "$@"
