#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"; export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
GUARD=$REPO/runs/benchmark_cdc/_guard; mkdir -p "$GUARD"; rm -f "$GUARD/grade_log.jsonl"
cd "$AB"; PORT=8811
"$PY" grading_mcp.py --problem fc46 --port $PORT --logdir "$GUARD" --backend thread --max-concurrent 4 \
  > "$GUARD/grader.log" 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null || true" EXIT
for i in $(seq 1 60); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 1; done
cat > "$GUARD/prompt.txt" <<'PROMPT'
This is a connectivity test. You have two MCP tools: grade_fast and grade_full.
Call grade_fast exactly once, with the argument `solution` set to this exact string:
int main(){return 0;}
Then report, verbatim, the JSON object the tool returned (its score and valid fields).
Do NOT write any files, do NOT spawn sub-agents, do NOT do anything else. Just that one tool call, then finish.
PROMPT
echo "[guard] launching codex (short)..."
timeout 360 codex exec -m gpt-5.6-sol -c model_reasoning_effort=high \
  --enable multi_agent_v2 -c features.multi_agent_v2.max_concurrent_threads_per_session=4 \
  -s workspace-write -c approval_policy=never --json -o "$GUARD/final.txt" -C "$GUARD" \
  -c "mcp_servers.grader.url=\"http://127.0.0.1:$PORT/mcp\"" \
  -c mcp_servers.grader.tool_timeout_sec=1200 -c mcp_servers.grader.startup_timeout_sec=60 \
  -c 'mcp_servers.grader.default_tools_approval_mode="approve"' \
  < "$GUARD/prompt.txt" > "$GUARD/codex_stdout.jsonl" 2>&1 || echo "[guard] codex exit=$? (timeout ok)"
echo "=== grade_log.jsonl (did codex call the tool?) ==="; cat "$GUARD/grade_log.jsonl" 2>/dev/null || echo "NONE"
echo "=== final.txt (agent's report) ==="; tail -c 600 "$GUARD/final.txt" 2>/dev/null
echo "=== grader.log tail ==="; tail -3 "$GUARD/grader.log"
