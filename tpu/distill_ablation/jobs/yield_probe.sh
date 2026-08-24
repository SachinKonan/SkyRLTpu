#!/bin/bash
# Measure ACTUAL rollout yield at candidate launch settings, and whether a shorter wall helps
# (short probe tasks never drop; long ones do -- so finishing sooner may beat the idle timeout).
set -uo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/yield_probe; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
CH="$WD/ch"; mkdir -p "$CH"; cp "$HOME/.codex/auth.json" "$CH/"
printf '[features]\nuse_legacy_landlock = true\n' > "$CH/config.toml"
P=$(ls -t /n/fs/vision-mix/sk7524/SkyRLTpu/runs/sweep/fc46_*/r1_s0_k0/prompt.txt | head -1)

run_batch () {   # $1=label $2=wall $3=count
  local L=$1 W=$2 N=$3
  for i in $(seq 1 $N); do
    ( mkdir -p "$WD/${L}_$i"
      CODEX_HOME="$CH" timeout $((W+60)) codex exec -m gpt-5.4-mini -c model_reasoning_effort=high \
        -s workspace-write -c approval_policy=never --json \
        -C "$WD/${L}_$i" -o "$WD/${L}_$i/final.txt" - < "$P" > "$WD/${L}_$i.jsonl" 2>&1 ) &
  done
  wait
  local d=0 f=0
  for i in $(seq 1 $N); do
    grep -q "idle timeout\|stream disconnected" "$WD/${L}_$i.jsonl" 2>/dev/null && d=$((d+1))
    [ -s "$WD/${L}_$i/final.txt" ] && f=$((f+1))
  done
  echo "YIELD label=$L concurrency=$N wall=${W}s drops=$d usable_final_msg=$f  ($((100*f/N))%)"
}

run_batch wall420 420 16
run_batch wall900 900 16
