#!/bin/bash
#SBATCH -p pli
#SBATCH --account=llm_explore
#SBATCH --qos=pli-low
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 60
#SBATCH --mem=300G
#SBATCH -t 1:00:00
#SBATCH -J ttd-ray-graders
#SBATCH -o /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-ttd-discover/runs/ray_graders/ray_head_%j.out
#SBATCH -e /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-ttd-discover/runs/ray_graders/ray_head_%j.out
#
# One fat cpu-node allocation running a Ray head, so the discover client (on the
# login node, where Tinker is reachable) can dispatch program grading as small
# Ray remotes across these 96 cores. Ray runs on the COMPUTE node (allowed),
# never the login node.
set -x
REPO=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-ttd-discover
VENV="${REPO}/third_party/discover/.venv-ttd-discover"
RUN=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-ttd-discover/runs/ray_graders
ADDR_FILE="${RUN}/ray_head.addr"
export RAY_TMPDIR="${RUN}/ray_tmp_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}" "${RUN}"
rm -f "${ADDR_FILE}"

source "${VENV}/bin/activate"

head_ip="$(hostname -i | awk '{print $1}')"
ray stop --force 2>/dev/null || true
ray start --head \
  --node-ip-address="${head_ip}" \
  --port=6379 \
  --num-cpus=60 \
  --temp-dir="${RAY_TMPDIR}" \
  --include-dashboard=false

printf '%s:6379\n' "${head_ip}" > "${ADDR_FILE}"
echo "RAY_HEAD_READY ${head_ip}:6379 job=${SLURM_JOB_ID} node=$(hostname)"

# Hold the allocation until walltime (SLURM kills at 2h); refresh a heartbeat.
end=$(( $(date +%s) + 7080 ))
while [ "$(date +%s)" -lt "$end" ]; do
  ray status >/dev/null 2>&1 || { echo "ray head died"; break; }
  sleep 30
done
ray stop --force 2>/dev/null || true
