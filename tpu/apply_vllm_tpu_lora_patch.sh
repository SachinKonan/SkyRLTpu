#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
PATCH_FILE="${PATCH_FILE:-${repo_root}/third_party/patches/tpu-inference-tpu-worker-lora-forwarders.patch}"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Patch file not found: ${PATCH_FILE}" >&2
  exit 1
fi

mapfile -t paths < <("${PYTHON}" - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("tpu_inference")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Could not find tpu_inference. Install vllm-tpu first.")

package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
print(package_dir.parent)
print(package_dir / "worker" / "tpu_worker.py")
PY
)

package_parent="${paths[0]}"
worker_path="${paths[1]}"

if [[ ! -f "${worker_path}" ]]; then
  echo "TPU worker file not found: ${worker_path}" >&2
  exit 1
fi

if grep -q "def add_lora(self, lora_request)" "${worker_path}"; then
  echo "vLLM TPU LoRA worker patch already present: ${worker_path}"
else
  if command -v patch >/dev/null 2>&1; then
    patch --forward -p0 -d "${package_parent}" < "${PATCH_FILE}"
  else
    "${PYTHON}" - "${worker_path}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = """    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

"""
insert = needle + """    def add_lora(self, lora_request) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

"""
if "def add_lora(self, lora_request)" in text:
    raise SystemExit(0)
if needle not in text:
    raise SystemExit("Could not find TPUWorker.get_supported_tasks patch anchor")
path.write_text(text.replace(needle, insert, 1))
PY
  fi
fi

"${PYTHON}" -m py_compile "${worker_path}"
"${PYTHON}" - <<'PY'
from tpu_inference.worker.tpu_worker import TPUWorker

missing = [
    name
    for name in ("add_lora", "remove_lora", "list_loras", "pin_lora")
    if not callable(getattr(TPUWorker, name, None))
]
if missing:
    raise SystemExit(f"TPUWorker missing LoRA methods after patch: {missing}")
PY

echo "vLLM TPU LoRA worker patch verified: ${worker_path}"
