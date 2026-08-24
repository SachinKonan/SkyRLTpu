#!/bin/bash
# GUARD: confirm which config path sets the SUBAGENT model, and that subagents actually run the
# cheap model while the orchestrator runs the expensive one. Prevents silently running 256
# rollouts on sol-xhigh. Trivial task, ~1 min.
set -euo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/subagent_guard; mkdir -p "$WD"; cd "$WD"

run_case () {
  local name="$1"; shift
  echo "=== case: $name"
  timeout 300 codex exec -m gpt-5.6-sol -c model_reasoning_effort=low \
     --enable multi_agent_v2 \
     -c features.multi_agent_v2.max_concurrent_threads_per_session=2 \
     "$@" \
     -s workspace-write -c approval_policy=never --json -C "$WD" \
     > "$WD/$name.jsonl" 2>&1 <<'PROMPT' || true
Spawn exactly ONE subagent using your multi-agent tool. Instruct that subagent to reply with only the word PONG. Then stop. Do not do anything else.
PROMPT
  echo "  models seen in event stream:"
  grep -oE '"model":"[^"]+"' "$WD/$name.jsonl" 2>/dev/null | sort | uniq -c | sed 's/^/    /' || echo "    (none)"
  grep -oiE 'unknown field[^,"]*|error loading config[^,"]*' "$WD/$name.jsonl" 2>/dev/null | head -2 | sed 's/^/    ERR: /' || true
  echo "  spawn/thread events: $(grep -c 'thread.started' "$WD/$name.jsonl" 2>/dev/null || echo 0) started"
}

run_case topLevel   -c default_subagent_model="gpt-5.4-mini" -c default_subagent_reasoning_effort="low"
run_case featuresNS -c features.multi_agent_v2.default_subagent_model="gpt-5.4-mini"

echo; echo "=== raw model/agent fields (first case) ==="
grep -oE '"(model|agent_role|agent_nickname|subagent_kind)":"[^"]*"' "$WD/topLevel.jsonl" 2>/dev/null | sort -u | head -20
