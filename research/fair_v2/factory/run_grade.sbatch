#!/bin/bash
#SBATCH -J fair-factory
#SBATCH -A zhuangl
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:59:00
#SBATCH -o /scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/slurm/grade-%j.out
# usage: sbatch run_grade.sbatch <candidate.py>:<screen|full>[:<out dir>] [...]
# results: <out dir>/result.json, default fair-data/factory/runs/<candidate stem>_<mode>/
# (resumable: rerun skips finished parts)
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2
nvidia-smi --query-gpu=name --format=csv,noheader
for spec in "$@"; do
  IFS=: read -r cand mode out <<< "$spec"
  out=${out:-/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/runs/$(basename "$cand" .py)_$mode}
  /usr/bin/time -f "wall %e s" /scratch/gpfs/ZHUANGL/sk7524/fair-env/bin/python factory/grade.py "$cand" --mode "$mode" \
    --out "$out" 2>&1 | grep -v Warning
done
