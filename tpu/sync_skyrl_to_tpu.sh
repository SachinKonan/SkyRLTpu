#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-tinker-v5p64-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER}/SkyRLTpu}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
git_cmd=(git --git-dir="${repo_root}/.git" --work-tree="${repo_root}")

commit="$("${git_cmd[@]}" rev-parse HEAD)"
mapfile -t submodule_paths < <("${git_cmd[@]}" config --file "${repo_root}/.gitmodules" --get-regexp '^submodule\..*\.path$' | awk '{print $2}')

if ! "${git_cmd[@]}" diff --quiet || ! "${git_cmd[@]}" diff --cached --quiet; then
  echo "Refusing to sync a dirty SkyRLTpu checkout. Commit or stash changes first." >&2
  exit 1
fi

declare -A submodule_commits=()
for submodule_path in "${submodule_paths[@]}"; do
  submodule_commit="$("${git_cmd[@]}" ls-tree HEAD "$submodule_path" | awk '{print $3}')"
  if [[ -z "$submodule_commit" ]]; then
    echo "Could not find $submodule_path in SkyRLTpu HEAD." >&2
    exit 1
  fi
  if ! git -C "${repo_root}/${submodule_path}" cat-file -e "${submodule_commit}^{commit}" 2>/dev/null; then
    echo "Missing $submodule_path commit $submodule_commit. Run: git submodule update --init $submodule_path" >&2
    exit 1
  fi
  submodule_commits["$submodule_path"]="$submodule_commit"
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

archive="${tmpdir}/SkyRLTpu-${commit}.tar.gz"
archive_tar="${tmpdir}/SkyRLTpu-${commit}.tar"
"${git_cmd[@]}" archive --format=tar HEAD > "$archive_tar"
for submodule_path in "${submodule_paths[@]}"; do
  submodule_commit="${submodule_commits[$submodule_path]}"
  submodule_tar="${tmpdir}/$(echo "$submodule_path" | tr / _)-${submodule_commit}.tar"
  git -C "${repo_root}/${submodule_path}" archive --format=tar --prefix="${submodule_path}/" "$submodule_commit" > "$submodule_tar"
  tar --concatenate --file "$archive_tar" "$submodule_tar"
done
gzip -c "$archive_tar" > "$archive"

remote_archive="/tmp/SkyRLTpu-${commit}.tar.gz"
remote_extract_cmd="
set -euo pipefail
mkdir -p \"$(dirname "${REMOTE_SKYRL_DIR}")\"
old_dir=\"${REMOTE_SKYRL_DIR}.old.\$(date +%s).\$\$\"
if [ -e '${REMOTE_SKYRL_DIR}' ]; then
  mv '${REMOTE_SKYRL_DIR}' \"\${old_dir}\"
fi
mkdir -p '${REMOTE_SKYRL_DIR}'
tar -xzf '${remote_archive}' -C '${REMOTE_SKYRL_DIR}'
printf '%s\n' '${commit}' > '${REMOTE_SKYRL_DIR}/.skyrltpu_commit'
rm -f '${remote_archive}'
if [ -n \"\${old_dir:-}\" ] && [ -e \"\${old_dir}\" ]; then
  (rm -rf \"\${old_dir}\" >/tmp/skyrltpu-sync-rm.log 2>&1 || true) &
fi
"

for ((worker = 0; worker < NUM_PROCESSES; worker++)); do
  echo "Syncing SkyRLTpu ${commit} to worker ${worker}:${REMOTE_SKYRL_DIR}"
  gcloud alpha compute tpus tpu-vm scp "$archive" "${REMOTE_USER}@${TPU_NAME}:${remote_archive}" \
    --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet

  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
    --command "$remote_extract_cmd"
done

echo "Synced SkyRLTpu ${commit} with ${#submodule_paths[@]} submodule(s) to ${NUM_PROCESSES} TPU workers."
