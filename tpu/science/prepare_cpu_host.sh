#!/usr/bin/env bash
# Trusted worker preparation. No candidate time is charged for fixed dependencies.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$(pwd)
mkdir -p .science
exec 9>.science/setup.lock
flock 9
sudo -n apt-get update -qq
sudo -n apt-get install -y -qq bubblewrap build-essential pkg-config libssl-dev
"$HOME/.local/bin/uv" --no-config venv --python 3.11.13 .science/venv --allow-existing
"$HOME/.local/bin/uv" --no-config pip sync --python .science/venv/bin/python tpu/science/requirements-cpu.lock
.science/venv/bin/python -c 'import numpy,scipy,pandas,statsmodels,sklearn,xgboost,cvxpy,matplotlib,sympy,qiskit; print("CPU solvers",cvxpy.installed_solvers())'
export CARGO_HOME="$root/.science/cargo"
export RUSTUP_HOME="$root/.science/rustup"
export CARGO_TARGET_DIR="$root/.science/router-target"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc
export PYO3_PYTHON="$root/.science/venv/bin/python"
export PATH="$CARGO_HOME/bin:$PATH"
taskset -c 16-19 cargo build --manifest-path .science/routing-task/rust/Cargo.toml --offline --locked --release -j4 -p router_cli
.science/venv/bin/python - <<'PY'
import hashlib,json,platform,time
from pathlib import Path
p=Path('.science/ready.json')
p.write_text(json.dumps(dict(python=platform.python_version(),time=time.time(),
    lock_sha256=hashlib.sha256(Path('tpu/science/requirements-cpu.lock').read_bytes()).hexdigest())))
PY
