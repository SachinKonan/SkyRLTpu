#!/bin/bash
# GUARD v3: correct key path -> does the subagent ACTUALLY run gpt-5.4-mini?
set -euo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/subagent_guard3; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
S="$HOME/.codex/sessions"; BEF=$(mktemp); find "$S" -maxdepth 5 -name '*.jsonl' -newermt '-1 day' 2>/dev/null | sort > "$BEF"
timeout 600 codex exec -m gpt-5.6-sol -c model_reasoning_effort=high \
  --enable multi_agent_v2 \
  -c features.multi_agent_v2.max_concurrent_threads_per_session=4 \
  -c features.multi_agent_v2.default_subagent_model="gpt-5.4-mini" \
  -c features.multi_agent_v2.default_subagent_reasoning_effort="low" \
  -s workspace-write -c approval_policy=never --json -C "$WD" \
  "Use your multi-agent capability to spawn TWO subagents; each should reply with the word PONG. Wait for both, then report." > B.jsonl 2>&1 || true
echo "collab events: $(grep -c collab_tool_call B.jsonl || echo 0)"
echo "final: $(grep -oE '"text":"[^"]{0,70}' B.jsonl | tail -1)"
AFT=$(mktemp); find "$S" -maxdepth 5 -name '*.jsonl' -newermt '-1 day' 2>/dev/null | sort > "$AFT"
echo "=== models per NEW session file ==="
comm -13 "$BEF" "$AFT" | while read -r f; do
  echo "-- $(basename "$f" | cut -c1-40)"
  grep -oE '"model":"[^"]+"' "$f" | sort -u | sed 's/^/     /'
  grep -oE '"reasoning_effort":"[^"]+"' "$f" | sort -u | head -1 | sed 's/^/     /'
done
