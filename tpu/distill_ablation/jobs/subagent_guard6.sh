#!/bin/bash
# GUARD v6: with expose_spawn_agent_model_overrides=true, can the ORCHESTRATOR pick the subagent
# model at spawn time? Checks (a) the spawn tool's arguments, (b) the real model in the session store.
set -euo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/subagent_guard6; rm -rf "$WD"; mkdir -p "$WD"
CH="$WD/codex_home"; mkdir -p "$CH"; cp "$HOME/.codex/auth.json" "$CH/"
cat > "$CH/config.toml" <<'TOML'
[features.multi_agent_v2]
enabled = true
expose_spawn_agent_model_overrides = true
max_concurrent_threads_per_session = 4
TOML
cd "$WD"
CODEX_HOME="$CH" timeout 900 codex exec --strict-config -m gpt-5.6-sol \
  -c model_reasoning_effort=high -s workspace-write -c approval_policy=never --json -C "$WD" \
  "First, describe the exact parameters your spawn-agent tool accepts (list every field name). Then spawn TWO subagents that each run the model gpt-5.4-mini -- pass the model explicitly if your tool supports it -- and have each reply with the word PONG. Wait for both, then report what model you asked them to use." > B.jsonl 2>&1 || true

echo "config-error: $(grep -ciE 'error loading config' B.jsonl || echo 0)"
echo "collab events: $(grep -c collab_tool_call B.jsonl || echo 0)"
echo "=== spawn tool call payloads (look for a model field) ==="
grep -o '"collab_tool_call"[^}]*}[^}]*}' B.jsonl 2>/dev/null | head -4 | cut -c1-400
echo "=== does any event mention a model override? ==="
grep -oiE '"model"[^,}]{0,40}|gpt-5\.4-mini' B.jsonl | sort | uniq -c | head -8
echo "=== agent's description of tool params ==="
grep -oE '"text":"[^"]{0,900}' B.jsonl | head -2 | cut -c1-900
echo "=== ACTUAL models in isolated session store ==="
find "$CH/sessions" -name '*.jsonl' 2>/dev/null | while read -r f; do
  echo "-- $(basename "$f" | cut -c1-40)"
  grep -oE '"model":"[^"]+"' "$f" | sort -u | sed 's/^/     /'
done
