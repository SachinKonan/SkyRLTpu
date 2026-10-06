#!/usr/bin/env bash
# Run a recursive GCS sync across both new and old Cloud SDK installations.
set -euo pipefail

gcloud_cli="${GCLOUD_STORAGE_CLI:-$(command -v gcloud || true)}"
if [[ -z "$gcloud_cli" ]]; then
  for candidate in "$HOME/google-cloud-sdk/bin/gcloud" /snap/bin/gcloud; do
    if [[ -x "$candidate" ]]; then
      gcloud_cli="$candidate"
      break
    fi
  done
fi
if [[ -n "$gcloud_cli" ]] &&
   "$gcloud_cli" storage rsync --help >/dev/null 2>&1; then
  exec "$gcloud_cli" storage rsync "$@"
fi

gsutil_cli="${GSUTIL_CLI:-$(command -v gsutil || true)}"
if [[ -z "$gsutil_cli" ]]; then
  for candidate in "$HOME/google-cloud-sdk/bin/gsutil" /snap/bin/gsutil; do
    if [[ -x "$candidate" ]]; then
      gsutil_cli="$candidate"
      break
    fi
  done
fi
if [[ -z "$gsutil_cli" ]]; then
  echo "neither gcloud storage rsync nor gsutil is available" >&2
  exit 127
fi

# gcloud and gsutil spell the exclusion flag differently.  Normalize the
# small shared argument surface before falling back to older Cloud SDKs.
gsutil_args=()
while (($#)); do
  case "$1" in
    --exclude=*)
      gsutil_args+=( -x "${1#*=}" )
      ;;
    --exclude)
      shift
      (($#)) || { echo "--exclude requires a pattern" >&2; exit 2; }
      gsutil_args+=( -x "$1" )
      ;;
    *)
      gsutil_args+=( "$1" )
      ;;
  esac
  shift
done
exec "$gsutil_cli" -m rsync "${gsutil_args[@]}"
