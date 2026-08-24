#!/bin/bash
# Do websocket drops depend on reasoning effort? Same REAL rollout prompt, 4 sessions each.
set -uo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/effort_probe; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
CH="$WD/ch"; mkdir -p "$CH"; cp "$HOME/.codex/auth.json" "$CH/"
printf '[features]\nuse_legacy_landlock = true\n' > "$CH/config.toml"
P=$(ls -t /n/fs/vision-mix/sk7524/SkyRLTpu/runs/sweep/fc46_*/r1_s0_k0/prompt.txt 2>/dev/null | head -1)
echo "using prompt: $P ($(wc -c < "$P") chars)"
for E in high medium; do
  for i in 1 2 3 4; do
    ( mkdir -p "$WD/${E}_$i"
      CODEX_HOME="$CH" timeout 700 codex exec -m gpt-5.4-mini -c model_reasoning_effort=$E \
        -s workspace-write -c approval_policy=never --json -C "$WD/${E}_$i" \
        -o "$WD/${E}_$i/final.txt" - < "$P" > "$WD/${E}_$i.jsonl" 2>&1 ) &
  done
done
wait
for E in high medium; do
  d=0; f=0
  for i in 1 2 3 4; do
    grep -q "idle timeout\|stream disconnected" "$WD/${E}_$i.jsonl" 2>/dev/null && d=$((d+1))
    [ -s "$WD/${E}_$i/final.txt" ] && f=$((f+1))
  done
  echo "EFFORT=$E sessions=4 ws_drops=$d final_msg_produced=$f"
done
