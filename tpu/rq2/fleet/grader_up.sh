#!/usr/bin/env bash
# Bring up distributed grading on ONE llama-farm slice, following the league's proven recipe
# (bringup_v5p64_league.sh): minimal grader venvs on every worker, Ray head on worker 0,
# workers joining over the internal VPC -- plus our HTTP shim on w0:8002 as the neuronic-facing
# ingress (clients cannot be cross-network Ray drivers through the 8001-8012 firewall window).
#
#   ./grader_up.sh <slice-name>          # e.g. sk7524-llamafarm-a-v5p64-east5a
#
# Worker 0 additionally gets the grading code bundle (league discover subset -- the ONLY tree
# with the payload-mode evaluator -- plus the frontiercs frontier_algo env for fc46/fc159) and
# a full evaluator venv. Idempotent; safe to re-run after preemption.
set -uo pipefail

SLICE="${1:?usage: grader_up.sh <slice>}"
VM="${SLICE}_spot"
case "$SLICE" in *east5b*) ZONE=us-east5-b;; *east5c*) ZONE=us-east5-c;; *) ZONE=us-east5-a;; esac
PROJECT=vision-mix
USER_AT=sk7524_princeton_edu
KEY="$HOME/.ssh/jobman_tpu_ed25519"
RQ2=/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq2
LEAGUE=/n/fs/vision-mix/sk7524/SkyRLTpu-league/third_party/discover
FRONTIER=/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover
BUCKET=gs://sk7524-tinker-tpu-us-east5
BUNDLE=$BUCKET/rq2-grader-bundle.tgz
RAY_PORT=6379
GRADE_PORT=8002
# league recipe: 150 cpus/worker on w1-7; w0 also serves vLLM + the shim, so fewer there
W0_CPUS="${W0_CPUS:-80}"
W_CPUS="${W_CPUS:-150}"

vssh() { local w="$1"; shift
  timeout "${SSH_TIMEOUT:-900}" gcloud alpha compute tpus tpu-vm ssh "${USER_AT}@${VM}" \
    --zone=$ZONE --project=$PROJECT --worker="$w" --ssh-key-file="$KEY" --command="$*" 2>&1 \
    | grep -viE "^warning|Attempting to|batch size"; }

log() { echo "[grader $(date '+%T')] $*"; }

# ---- 0. bundle: build once, reuse until sources change --------------------------------------
if ! gsutil -q stat "$BUNDLE" 2>/dev/null || [[ "${REBUNDLE:-0}" == "1" ]]; then
  log "building grader bundle -> $BUNDLE"
  T=$(mktemp -d)
  mkdir -p "$T/b"
  # league discover subset: the payload-mode evaluator + envs + package plumbing
  rsync -a --exclude='.venv*' --exclude='__pycache__' --exclude='.git' \
        --exclude='runs' --exclude='logs' "$LEAGUE/" "$T/b/discover/"
  # frontiercs: just the frontier_algo example (env, judge, testlib, problems w/ testdata)
  mkdir -p "$T/b/frontiercs/examples"
  rsync -a --exclude='__pycache__' "$FRONTIER/examples/frontier_algo" "$T/b/frontiercs/examples/"
  cp "$RQ2/fleet/grade_core.py" "$RQ2/fleet/grade_shim.py" "$T/b/"
  tar czf "$T/bundle.tgz" -C "$T/b" .
  gsutil -q cp "$T/bundle.tgz" "$BUNDLE"
  rm -rf "$T"
  log "bundle uploaded ($(gsutil du -s "$BUNDLE" 2>/dev/null | awk '{print int($1/1048576)"MB"}'))"
fi

# ---- 1. worker IPs (internal for ray, we ssh by worker index) -------------------------------
W0INT=$(timeout 120 gcloud compute tpus tpu-vm describe "$VM" --zone=$ZONE --project=$PROJECT \
  --format="value(networkEndpoints[0].ipAddress)" 2>/dev/null)
[[ -n "$W0INT" ]] || { log "cannot resolve worker0 internal IP"; exit 1; }
log "$SLICE: w0 internal $W0INT"

# ---- 2. grader venvs + ray workers on w1-7 (league recipe, in parallel) ---------------------
RAYV="${RAYV:-2.48.0}"
for w in 1 2 3 4 5 6 7; do
  ( vssh "$w" "
      export PATH=\$HOME/.local/bin:\$PATH
      command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
      export PATH=\$HOME/.local/bin:\$PATH
      [ -x ~/.venvs/grader/bin/ray ] || {
        uv venv ~/.venvs/grader --python 3.11 >/dev/null 2>&1
        uv pip install --python ~/.venvs/grader/bin/python 'ray==$RAYV' numpy scipy shapely numba scikit-learn psutil >/dev/null 2>&1
      }
      ~/.venvs/grader/bin/ray stop >/dev/null 2>&1; sleep 2
      ~/.venvs/grader/bin/ray start --address=$W0INT:$RAY_PORT --num-cpus=$W_CPUS --disable-usage-stats >/tmp/ray-worker.log 2>&1 \
        && echo RAY-WORKER-\$(hostname)
    " ) &
done

# ---- 3. w0: evaluator venv + bundle + ray head + shim ---------------------------------------
vssh 0 "
  set -e
  export PATH=\$HOME/.local/bin:\$PATH
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH=\$HOME/.local/bin:\$PATH
  mkdir -p ~/grader && cd ~/grader
  # bundle (idempotent on same generation: gsutil cp only when missing or forced)
  [ -d discover ] || { gsutil -q cp $BUNDLE bundle.tgz && tar xzf bundle.tgz && rm -f bundle.tgz; }
  # evaluator venv: league's uv sync recipe against the bundled discover tree
  if [ ! -x ~/grader/discover/.venv/bin/python ]; then
    cd ~/grader/discover && uv sync --extra math --python 3.11 >~/grader/venv-build.log 2>&1 || true
    cd ~/grader
  fi
  PY=~/grader/discover/.venv/bin/python
  \$PY -c 'import ray, numpy' 2>/dev/null || { echo VENV-BAD; exit 1; }
  \$PY -m ray.scripts.scripts stop >/dev/null 2>&1 || true; sleep 2
  ~/grader/discover/.venv/bin/ray start --head --port=$RAY_PORT --num-cpus=$W0_CPUS \
    --disable-usage-stats >/tmp/ray-head.log 2>&1 && echo RAY-HEAD-UP
  tmux kill-session -t grade-shim 2>/dev/null || true
  tmux new-session -d -s grade-shim \
    \"cd ~/grader && GRADER_HOME=\$HOME/grader RAY_ADDRESS=127.0.0.1:$RAY_PORT \
      GRADE_PORT=$GRADE_PORT GRADE_SLOTS=600 TTD_EVAL_BACKEND=ray TTD_RAY_PAYLOAD=1 \
      TTD_DISCOVER_SYNC=0 PYTHONPATH=\$HOME/grader \
      ~/grader/discover/.venv/bin/python grade_shim.py 2>&1 | tee ~/grader/shim.log\"
  sleep 3
  curl -sf -m 8 http://127.0.0.1:$GRADE_PORT/health && echo && echo SHIM-UP
"
wait
log "$SLICE grader bring-up finished"
