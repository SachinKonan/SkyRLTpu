#!/bin/bash
# GUARD v5: valid config.toml (table only, enabled=true) -> does the subagent run gpt-5.4-mini?
set -euo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/subagent_guard5; rm -rf "$WD"; mkdir -p "$WD"
CH="$WD/codex_home"; mkdir -p "$CH"; cp "$HOME/.codex/auth.json" "$CH/"
cat > "$CH/config.toml" <<'TOML'
[features.multi_agent_v2]
enabled = true
default_subagent_model = "gpt-5.4-mini"
default_subagent_reasoning_effort = "low"
max_concurrent_threads_per_session = 4
expose_spawn_agent_model_overrides = true
TOML
cd "$WD"
CODEX_HOME="$CH" timeout 600 codex exec --strict-config -m gpt-5.6-sol \
  -c model_reasoning_effort=high -s workspace-write -c approval_policy=never --json -C "$WD" \
  "Use your multi-agent capability to spawn TWO subagents; each replies with the word PONG. Wait for both, then report." > B.jsonl 2>&1 || true
echo "config-error: $(grep -ciE 'error loading config' B.jsonl || echo 0)"
echo "collab events: $(grep -c collab_tool_call B.jsonl || echo 0)"
echo "final: $(grep -oE '\"text\":\"[^\"]{0,60}' B.jsonl | tail -1)"
echo "=== models in isolated store ==="
find "$CH/sessions" -name '*.jsonl' 2>/dev/null | while read -r f; do
  echo "-- $(basename "$f" | cut -c1-40)"
  grep -oE '"model":"[^"]+"' "$f" | sort -u | sed 's/^/     /'
done
