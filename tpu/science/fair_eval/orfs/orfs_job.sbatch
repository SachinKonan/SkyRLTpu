#!/bin/bash
# One Tier-2 OpenROAD run per job: 16 cores (evaluate_with_orfs.py passes -threads 16, matching the
# judges' 16-core machine), up to 12 h (the script's own timeout). Submit with
#   sbatch --export=ALL,NAME=...,DESIGN=...,POS=... -J orfs-NAME-DESIGN orfs_job.sbatch
#SBATCH --partition=cpu
#SBATCH --account=zhuangl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:30:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval/orfs/slurm/%x-%j.out
exec bash /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-momo-placement/tpu/science/fair_eval/orfs/run_orfs.sh "$NAME" "$DESIGN" "$POS"
