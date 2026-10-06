#!/usr/bin/env bash
# jobman setup for ONE meta-tree generation on a v5p-128 (16 hosts).
# Three members, each an independent single-model cell on its own 4-host block,
# plus 4 spare hosts assigned to SPARE_FOR as extra vLLM replicas:
#   qwen  w0-3   gemma  w4-7   muse  w8-11   spares w12-15
# w0 orchestrates: runs cell_worker.sh ON EACH TRAINER HOST (w0 local, w4/w8
# over ssh) with that member's IP sublist -- so each member gets the exact
# engine recipe its stage cell validated, including muse's tuned m-* case.
# Bring-ups run in PARALLEL (disjoint hosts; serial would cost ~3h).
set -euo pipefail
: "${JOBMAN_WORKER_ID:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"; : "${ARM:?}"; : "${GEN:?}"
[ "$JOBMAN_WORKER_ID" = "0" ] || { echo "worker $JOBMAN_WORKER_ID: driven from w0"; exit 0; }

export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/SkyRLTpu-league"
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
INT="$JOBMAN_TPU_INTERNAL_IPS"
ln -sfn "$REPO" "$HOME/ttd-client"

ip_at() { echo "$INT" | cut -d, -f"$1"; }
# Layout: 1-based worker-field lists per member (trainer FIRST), overridable so
# one script serves both shapes:
#   v5p-128 (default): qwen 1-4 (+spares 13-16), gemma 5-8, muse 9-12
#   v5p-64 compressed: FIELDS_QWEN="1,2,7" FIELDS_GEMMA="3,4" FIELDS_MUSE="5,6"
#                      GRADER_FIELD=8 GRADER_FOR=gemma
SPARE_FOR="${SPARE_FOR:-qwen}"
FIELDS_QWEN="${FIELDS_QWEN:-1,2,3,4}"
FIELDS_GEMMA="${FIELDS_GEMMA:-5,6,7,8}"
FIELDS_MUSE="${FIELDS_MUSE:-9,10,11,12}"
SPARE_FIELDS="${SPARE_FIELDS:-13,14,15,16}"
GRADER_FIELD="${GRADER_FIELD:-}"
GRADER_FOR="${GRADER_FOR:-gemma}"

# field lists accept ',' or ';' separators (';' lets them ride inside the
# comma-separated META_LAYOUT_ENV passthrough)
fields_to_ips() { local out="" f; for f in $(echo "$1" | tr ',;' '  '); do out="$out,$(ip_at "$f")"; done; echo "${out#,}"; }

member_ips() {  # $1 = tag -> comma IP list, trainer first
  local base
  case "$1" in
    qwen)  base=$(fields_to_ips "$FIELDS_QWEN") ;;
    gemma) base=$(fields_to_ips "$FIELDS_GEMMA") ;;
    muse)  base=$(fields_to_ips "$FIELDS_MUSE") ;;
  esac
  if [ -n "$SPARE_FIELDS" ] && [ "$1" = "$SPARE_FOR" ]; then
    echo "$base,$(fields_to_ips "$SPARE_FIELDS")"
  else echo "$base"; fi
}
member_cell()  { case "$1" in qwen) echo "meta-qwen";; gemma) echo "g-meta";; muse) echo "m-meta";; esac; }
member_model() { case "$1" in qwen) echo "qwen3.5-27b Qwen/Qwen3.5-27B";; gemma) echo "gemma4-31b google/gemma-4-31B-it";; muse) echo "muse-glimmer-30b meta-models/Muse-Glimmer-30B";; esac; }

bring_up_member() {  # $1 = tag; runs cell_worker on the member's trainer host
  local tag="$1" ips trainer cell mm
  ips=$(member_ips "$tag"); trainer=${ips%%,*}; cell=$(member_cell "$tag")
  mm=$(member_model "$tag")
  local mtx=${mm%% *} hf=${mm##* }
  if [ "$trainer" = "$(ip_at 1)" ]; then
    TUNIX_MAXTEXT_MODEL_NAME="$mtx" MODEL_NAME="$hf" bash "$REPO/tpu/jobman/ensure_orbax_ckpt.sh" || true
    CELL="$cell" JOBMAN_WORKER_ID=0 JOBMAN_TPU_INTERNAL_IPS="$ips" \
      bash "$REPO/tpu/jobman/cell_worker.sh"
  else
    timeout 5400 ssh $SSHO sk7524_princeton_edu@"$trainer" "
      export PATH=\$HOME/.local/bin:\$PATH
      ln -sfn \$HOME/SkyRLTpu-league \$HOME/ttd-client
      TUNIX_MAXTEXT_MODEL_NAME='$mtx' MODEL_NAME='$hf' bash \$HOME/SkyRLTpu-league/tpu/jobman/ensure_orbax_ckpt.sh || true
      CELL='$cell' JOBMAN_WORKER_ID=0 JOBMAN_TPU_INTERNAL_IPS='$ips' \
        bash \$HOME/SkyRLTpu-league/tpu/jobman/cell_worker.sh"
  fi
}

pids=(); tags=(qwen gemma muse)
for t in "${tags[@]}"; do
  ( bring_up_member "$t" > ~/meta-bringup-$t.log 2>&1; echo $? > ~/meta-bringup-$t.rc ) &
  pids+=($!)
done
wait "${pids[@]}" || true
ok=1
for t in "${tags[@]}"; do
  rc=$(cat ~/meta-bringup-$t.rc 2>/dev/null || echo 1)
  if [ "$rc" != 0 ]; then
    echo "member $t bring-up FAILED (rc=$rc):"; tail -6 ~/meta-bringup-$t.log; ok=0
  else
    echo "member $t ready"
  fi
done
[ "$ok" = 1 ] || exit 1

# Optional dedicated grader host: joins ONE member's ray cluster (a ray node
# belongs to exactly one cluster). Measured grading hides inside sampling, so
# this is a luxury -- pinned to GRADER_FOR, cheap to repoint or reassign.
if [ -n "$GRADER_FIELD" ]; then
  ghost=$(fields_to_ips "$GRADER_FIELD")
  ghead=$(member_ips "$GRADER_FOR"); ghead=${ghead%%,*}
  RAYV=$("$REPO/third_party/discover/.venv-ttd-discover/bin/python" -c "import ray; print(ray.__version__)" 2>/dev/null || echo "2.30.0")
  timeout 900 ssh $SSHO sk7524_princeton_edu@"$ghost" "
    export PATH=\$HOME/.local/bin:\$PATH
    pgrep -f '[r]ay/core' >/dev/null && { echo 'grader ray already up'; exit 0; }
    [ -x ~/.venvs/grader/bin/ray ] || {
      uv venv ~/.venvs/grader --python 3.11 >/dev/null 2>&1
      uv pip install --python ~/.venvs/grader/bin/python 'ray==$RAYV' numpy scipy shapely numba scikit-learn psutil >/dev/null 2>&1
    }
    ~/.venvs/grader/bin/ray start --address=$ghead:6379 --num-cpus=150 --disable-usage-stats >/tmp/ray-worker.log 2>&1 && echo 'grader host joined $GRADER_FOR'
  " 2>/dev/null || echo "grader host join FAILED (grading degrades, not fatal)"
fi
echo "meta worker 0 ready ($ARM gen $GEN, spares -> $SPARE_FOR)"
