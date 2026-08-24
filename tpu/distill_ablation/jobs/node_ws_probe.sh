#!/bin/bash
# Is the websocket-disconnect failure node-specific? Run 4 short codex sessions and count drops.
set -x
set -uo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
N=$(hostname -s)
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ws_probe/$N; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
CH="$WD/ch"; mkdir -p "$CH"; cp "$HOME/.codex/auth.json" "$CH/"
printf '[features]\nuse_legacy_landlock = true\n' > "$CH/config.toml"
for i in 1 2 3 4; do
  ( CODEX_HOME="$CH" timeout 300 codex exec -m gpt-5.4-mini -c model_reasoning_effort=high \
      -s workspace-write -c approval_policy=never --json -C "$WD" \
      "Write a short C++ function that sorts a vector, then explain it in one sentence." \
      > "$WD/s$i.jsonl" 2>&1 || true ) &
done
wait
drops=$(grep -l "idle timeout waiting for websocket\|stream disconnected" "$WD"/s*.jsonl 2>/dev/null | wc -l)
done_ok=$(grep -l '"type":"turn.completed"' "$WD"/s*.jsonl 2>/dev/null | wc -l)
echo "NODE=$N sessions=4 websocket_drops=$drops completed=$done_ok"
