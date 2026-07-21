# Transport helpers for TPU-VM launcher scripts.
#
# Default transport is `gcloud alpha compute tpus tpu-vm ...`. Setting
# TPU_SSH_MODE=direct switches to plain ssh/scp against explicit worker IPs,
# for environments where gcloud CLI auth is unavailable but the VMs are
# directly reachable (e.g. ADC-only sessions):
#   TPU_SSH_MODE=direct
#   TPU_EXTERNAL_IPS=ip0,ip1,...   # worker-indexed, comma-separated
#   TPU_INTERNAL_IPS=ip0,ip1,...   # worker-indexed, comma-separated
# Callers must have REMOTE_USER, TPU_NAME, PROJECT, ZONE, SSH_KEY_FILE set.

_tpu_direct_ssh_opts=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o IdentitiesOnly=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
  -o ConnectTimeout=30
)

_tpu_direct_ext_ip() {
  local worker="$1"
  local -a ips
  IFS=',' read -r -a ips <<< "${TPU_EXTERNAL_IPS:?TPU_SSH_MODE=direct requires TPU_EXTERNAL_IPS}"
  if (( worker < 0 || worker >= ${#ips[@]} )); then
    echo "worker ${worker} out of range for TPU_EXTERNAL_IPS=${TPU_EXTERNAL_IPS}" >&2
    return 1
  fi
  echo "${ips[$worker]}"
}

# tpu_vm_ssh <worker> <remote_command>
tpu_vm_ssh() {
  local worker="$1" remote_cmd="$2"
  if [[ "${TPU_SSH_MODE:-gcloud}" == "direct" ]]; then
    local ip
    ip="$(_tpu_direct_ext_ip "$worker")" || return 1
    ssh -i "$SSH_KEY_FILE" "${_tpu_direct_ssh_opts[@]}" "${REMOTE_USER}@${ip}" "$remote_cmd"
  else
    gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
      --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
      --command "$remote_cmd"
  fi
}

# tpu_vm_scp <worker> <local_src> <remote_path>
tpu_vm_scp() {
  local worker="$1" src="$2" remote_path="$3"
  if [[ "${TPU_SSH_MODE:-gcloud}" == "direct" ]]; then
    local ip
    ip="$(_tpu_direct_ext_ip "$worker")" || return 1
    scp -i "$SSH_KEY_FILE" "${_tpu_direct_ssh_opts[@]}" "$src" "${REMOTE_USER}@${ip}:${remote_path}"
  else
    gcloud alpha compute tpus tpu-vm scp "$src" "${REMOTE_USER}@${TPU_NAME}:${remote_path}" \
      --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet
  fi
}

# tpu_vm_external_ip <worker>
tpu_vm_external_ip() {
  local worker="$1"
  if [[ "${TPU_SSH_MODE:-gcloud}" == "direct" ]]; then
    _tpu_direct_ext_ip "$worker"
  else
    gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --format="value(networkEndpoints[${worker}].accessConfig.externalIp)"
  fi
}

# tpu_vm_internal_ips  (semicolon-separated, matching gcloud value() output)
tpu_vm_internal_ips() {
  if [[ "${TPU_SSH_MODE:-gcloud}" == "direct" ]]; then
    echo "${TPU_INTERNAL_IPS:?TPU_SSH_MODE=direct requires TPU_INTERNAL_IPS}" | tr ',' ';'
  else
    gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --format='value(networkEndpoints.ipAddress)'
  fi
}
