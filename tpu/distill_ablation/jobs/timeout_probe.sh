#!/bin/bash
# Does raising stream_idle_timeout_ms stop the websocket drops?
# Baseline measured minutes ago on this same node/prompt/concurrency: 7/8 sessions dropped.
set -uo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/timeout_probe; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
CH="$WD/ch"; mkdir -p "$CH"; cp "$HOME/.codex/auth.json" "$CH/"
cat > "$CH/config.toml" <<'TOML'
[features]
use_legacy_landlock = true

[model_providers.openai]
stream_idle_timeout_ms = 1800000
stream_max_retries = 20
request_max_retries = 10

[model_providers.chatgpt]
stream_idle_timeout_ms = 1800000
stream_max_retries = 20
request_max_retries = 10
TOML
P=$(ls -t /n/fs/vision-mix/sk7524/SkyRLTpu/runs/sweep/fc46_*/r1_s0_k0/prompt.txt 2>/dev/null | head -1)
echo "prompt=$P ($(wc -c < "$P") chars), 8 concurrent, idle_timeout=30min"
for i in $(seq 1 8); do
  ( mkdir -p "$WD/s$i"
    CODEX_HOME="$CH" timeout 900 codex exec --strict-config -m gpt-5.4-mini \
      -c model_reasoning_effort=high -s workspace-write -c approval_policy=never --json \
      -C "$WD/s$i" -o "$WD/s$i/final.txt" - < "$P" > "$WD/s$i.jsonl" 2>&1 ) &
done
wait
d=0; f=0; cfg=0
for i in $(seq 1 8); do
  grep -q "idle timeout\|stream disconnected" "$WD/s$i.jsonl" 2>/dev/null && d=$((d+1))
  [ -s "$WD/s$i/final.txt" ] && f=$((f+1))
  grep -qi "error loading config" "$WD/s$i.jsonl" 2>/dev/null && cfg=$((cfg+1))
done
echo "RESULT sessions=8 ws_drops=$d final_msg=$f config_errors=$cfg   (baseline was drops=7 final=1)"
