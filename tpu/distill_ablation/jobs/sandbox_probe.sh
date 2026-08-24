#!/bin/bash
# Which sandbox mode lets a codex rollout actually RUN SHELL COMMANDS on a compute node?
# bwrap needs user namespaces, and max_user_namespaces=0 cluster-wide -> must find an alternative.
set -euo pipefail
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
WD=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/sandbox_probe; rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
echo "node=$(hostname)"
echo "max_user_namespaces=$(cat /proc/sys/user/max_user_namespaces 2>/dev/null)"

probe () {
  local name="$1"; shift
  local H="$WD/ch_$name"; mkdir -p "$H"; cp "$HOME/.codex/auth.json" "$H/"
  [ -n "${1:-}" ] && printf '%s\n' "$1" > "$H/config.toml" && shift
  mkdir -p "$WD/$name"
  CODEX_HOME="$H" timeout 240 codex exec -m gpt-5.4-mini -c model_reasoning_effort=low \
      "$@" --json -C "$WD/$name" \
      'Run exactly this shell command: echo SANDBOX_OK_12345 . Then report whether it succeeded.' \
      > "$WD/$name.jsonl" 2>&1 || true
  local ok=$(grep -c 'SANDBOX_OK_12345' "$WD/$name.jsonl" 2>/dev/null || echo 0)
  local bw=$(grep -ci 'bwrap' "$WD/$name.jsonl" 2>/dev/null || echo 0)
  echo "  $name: shell_worked=$([ "$ok" -gt 1 ] && echo YES || echo no) bwrap_errors=$bw"
}

probe A_workspace_write   ""                                        -s workspace-write -c approval_policy=never
probe B_bwrap_off         $'[features]\nuse_linux_sandbox_bwrap = false' -s workspace-write -c approval_policy=never
probe C_landlock          $'[features]\nuse_legacy_landlock = true'  -s workspace-write -c approval_policy=never
probe D_full_access       ""                                        -s danger-full-access -c approval_policy=never
