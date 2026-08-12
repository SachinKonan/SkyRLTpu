#!/bin/bash
# Slurm wrapper for the Muse-Glimmer end-to-end TPU proof.
#
# The orchestration itself (gcloud + ssh polling) is light, but it runs for up
# to three hours -- which makes it exactly the kind of thing that must live on a
# compute node rather than the login node.
#
# Signal handling is the whole point of this wrapper.  `--signal=TERM@300`
# (deliberately WITHOUT the `B:` prefix) tells slurm to signal every process in
# the step, so e2e_tpu.sh itself receives the TERM and runs its own trap, which
# issues a `setsid nohup` QR delete that survives the shell being killed.  Using
# `B:TERM` would deliver only to this wrapper shell and the QR would leak -- the
# exact failure this repo has been bitten by twice.
#
#SBATCH --job-name=mg-e2e
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/e2e-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
export PATH=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin:$HOME/.local/bin:$PATH
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

# Belt and braces: if the wrapper is torn down for any reason the child did not
# already handle, delete the QR anyway.  Idempotent -- a delete of an
# already-gone QR is a no-op.
QR="${QR:-sk7524-museglimmer-e2e}"
ZONE="${ZONE:-us-east5-a}"
backstop() {
  echo "=== wrapper backstop: ensuring ${QR} is gone ==="
  setsid nohup /n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud alpha compute tpus \
    queued-resources delete "$QR" --project=vision-mix --zone="$ZONE" --quiet --force \
    >>"${REPO}/runs/muse_glimmer/qr-delete.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
  sleep 20
}
trap 'backstop' EXIT TERM INT HUP

bash "${REPO}/tpu/muse_glimmer/e2e_tpu.sh"
rc=$?
echo "=== e2e_tpu.sh exit ${rc} $(date -Is) ==="

# Final verification, printed into the job log so the result is auditable
# without re-running anything.
for z in us-east5-a us-east5-b; do
  echo "--- muse QRs remaining in ${z} ---"
  timeout 120 /n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud compute tpus \
    queued-resources list --zone "$z" --format="value(name,state.state)" 2>/dev/null \
    | grep -i muse || echo "(none)"
done
exit $rc
