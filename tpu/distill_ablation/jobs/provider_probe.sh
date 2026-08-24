#!/bin/bash
# Can a CUSTOM provider (built-ins are locked) carry a long stream_idle_timeout_ms with ChatGPT
# auth, and does it stop the drops? Baseline on this node/prompt/concurrency: 7/8 dropped.
set -uo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/provider_probe; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
CH="$WD/ch"; mkdir -p "$CH"; cp "$HOME/.codex/auth.json" "$CH/"
cat > "$CH/config.toml" <<'TOML'
model_provider = "openai-long"

[features]
use_legacy_landlock = true

[model_providers.openai-long]
name = "OpenAI ChatGPT (long idle timeout)"
base_url = "https://chatgpt.com/backend-api/codex"
wire_api = "responses"
requires_openai_auth = true
stream_idle_timeout_ms = 1800000
stream_max_retries = 20
request_max_retries = 10
TOML

echo "### step 1: does auth work through the custom provider?"
CODEX_HOME="$CH" timeout 180 codex exec --strict-config -m gpt-5.4-mini \
  -c model_reasoning_effort=low --json -C "$WD" "Reply with only: OK" > auth.jsonl 2>&1 || true
if grep -qi "error loading config" auth.jsonl; then echo "  CONFIG ERROR: $(grep -i -A2 'error loading' auth.jsonl | head -3 | tr '\n' ' ')"; exit 0; fi
grep -qE '"text":"[^"]*OK' auth.jsonl && echo "  AUTH OK" || { echo "  AUTH FAILED: $(tail -c 300 auth.jsonl | tr '\n' ' ')"; exit 0; }

echo "### step 2: 8 concurrent long rollouts (baseline was 7/8 dropped)"
P=$(ls -t /n/fs/vision-mix/sk7524/SkyRLTpu/runs/sweep/fc46_*/r1_s0_k0/prompt.txt | head -1)
for i in $(seq 1 8); do
  ( mkdir -p "$WD/s$i"
    CODEX_HOME="$CH" timeout 960 codex exec --strict-config -m gpt-5.4-mini \
      -c model_reasoning_effort=high -s workspace-write -c approval_policy=never --json \
      -C "$WD/s$i" -o "$WD/s$i/final.txt" - < "$P" > "$WD/s$i.jsonl" 2>&1 ) &
done
wait
d=0; f=0
for i in $(seq 1 8); do
  grep -q "idle timeout\|stream disconnected" "$WD/s$i.jsonl" 2>/dev/null && d=$((d+1))
  [ -s "$WD/s$i/final.txt" ] && f=$((f+1))
done
echo "RESULT custom_provider sessions=8 drops=$d usable=$f  (baseline drops=7 usable=1)"
