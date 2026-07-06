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
jobman_path="third_party/jobman"
jobman_commit="$("${git_cmd[@]}" ls-tree HEAD "$jobman_path" | awk '{print $3}')"

if ! "${git_cmd[@]}" diff --quiet || ! "${git_cmd[@]}" diff --cached --quiet; then
  echo "Refusing to sync a dirty SkyRLTpu checkout. Commit or stash changes first." >&2
  exit 1
fi

if [[ -z "$jobman_commit" ]]; then
  echo "Could not find $jobman_path in SkyRLTpu HEAD." >&2
  exit 1
fi

if ! git -C "${repo_root}/${jobman_path}" cat-file -e "${jobman_commit}^{commit}" 2>/dev/null; then
  echo "Missing $jobman_path commit $jobman_commit. Run: git submodule update --init $jobman_path" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

archive="${tmpdir}/SkyRLTpu-${commit}.tar.gz"
archive_tar="${tmpdir}/SkyRLTpu-${commit}.tar"
jobman_tar="${tmpdir}/jobman-${jobman_commit}.tar"
"${git_cmd[@]}" archive --format=tar HEAD > "$archive_tar"
git -C "${repo_root}/${jobman_path}" archive --format=tar --prefix="${jobman_path}/" "$jobman_commit" > "$jobman_tar"
tar --concatenate --file "$archive_tar" "$jobman_tar"
gzip -c "$archive_tar" > "$archive"

remote_archive="/tmp/SkyRLTpu-${commit}.tar.gz"
remote_extract_cmd="
set -euo pipefail
mkdir -p \"$(dirname "${REMOTE_SKYRL_DIR}")\"
rm -rf '${REMOTE_SKYRL_DIR}'
mkdir -p '${REMOTE_SKYRL_DIR}'
tar -xzf '${remote_archive}' -C '${REMOTE_SKYRL_DIR}'
printf '%s\n' '${commit}' > '${REMOTE_SKYRL_DIR}/.skyrltpu_commit'
rm -f '${remote_archive}'
"

for ((worker = 0; worker < NUM_PROCESSES; worker++)); do
  echo "Syncing SkyRLTpu ${commit} to worker ${worker}:${REMOTE_SKYRL_DIR}"
  gcloud alpha compute tpus tpu-vm scp "$archive" "${REMOTE_USER}@${TPU_NAME}:${remote_archive}" \
    --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet

  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
    --command "$remote_extract_cmd"
done

echo "Synced SkyRLTpu ${commit} with jobman ${jobman_commit} to ${NUM_PROCESSES} TPU workers."
