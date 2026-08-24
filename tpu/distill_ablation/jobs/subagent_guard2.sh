#!/bin/bash
# GUARD v2: (1) does a plain session produce output at all? (2) does multi_agent_v2 actually spawn?
# (3) which model do subagents run? Check the codex session store, which records per-thread model.
set -euo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/subagent_guard2; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
SESS="$HOME/.codex/sessions"
BEFORE=$(mktemp); find "$SESS" -newermt '-1 day' -type f 2>/dev/null | sort > "$BEFORE" || true

echo "### A. plain session, positional prompt, effort=high"
timeout 300 codex exec -m gpt-5.6-sol -c model_reasoning_effort=high \
  -s workspace-write -c approval_policy=never --json -C "$WD" \
  "Reply with the single word PONG and nothing else." > A.jsonl 2>&1 || true
echo "  bytes=$(wc -c < A.jsonl)  agent_text: $(grep -oE '"text":"[^"]{0,60}' A.jsonl | tail -2 | tr '\n' ' ')"

echo "### B. multi_agent_v2 spawn, subagent model override (top-level key)"
timeout 600 codex exec -m gpt-5.6-sol -c model_reasoning_effort=high \
  --enable multi_agent_v2 \
  -c features.multi_agent_v2.max_concurrent_threads_per_session=2 \
  -c default_subagent_model="gpt-5.4-mini" \
  -c default_subagent_reasoning_effort="low" \
  -s workspace-write -c approval_policy=never --json -C "$WD" \
  "Use your multi-agent capability to spawn ONE subagent whose entire task is to reply with the word PONG. Wait for it, then tell me what it said." > B.jsonl 2>&1 || true
echo "  bytes=$(wc -c < B.jsonl)"
echo "  event types:"; grep -oE '"type":"[a-z_.]+"' B.jsonl | sort | uniq -c | sort -rn | head -10 | sed 's/^/    /'
echo "  collab/spawn hits: $(grep -oicE 'collab|spawn' B.jsonl | head -1)"
echo "  agent text tail: $(grep -oE '"text":"[^"]{0,80}' B.jsonl | tail -3 | tr '\n' ' ')"

echo "### C. new session files -> which models were used?"
AFTER=$(mktemp); find "$SESS" -newermt '-1 day' -type f 2>/dev/null | sort > "$AFTER" || true
NEW=$(comm -13 "$BEFORE" "$AFTER" | head -20)
echo "  new session files: $(echo "$NEW" | grep -c . )"
for f in $NEW; do
  echo "   -- $(basename $f)"
  grep -oE '"model":"[^"]+"' "$f" 2>/dev/null | sort | uniq -c | sed 's/^/      /' | head -5
  grep -oE '"(agent_role|subagent_kind|agent_nickname)":"[^"]*"' "$f" 2>/dev/null | sort -u | head -3 | sed 's/^/      /'
done
