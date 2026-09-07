# Frozen Client Environment

`uv.lock` freezes the full Linux x86-64 Python 3.12 client dependency graph.
Tinker 0.22.7 matches the frozen discover bundle's client lock; Ray 2.58.0,
Transformers 5.8.0 and CPU-only Torch 2.10.0 preserve the Ray launcher settings.

`Host.install_client` uses `uv sync --frozen`, then installs the bundled
discover source with `--no-deps --no-build-isolation`. Locked setuptools and
wheel handle that build. The environment marker includes the source bundle,
project and lock hashes; old unlocked environments are rebuilt on next setup.
The installed versions are checked against the lock before reuse or launch.
Changing files here does not modify already-running jobs.

To intentionally update dependencies, edit this project's requirements and run
`uv lock --project tpu/swarm/ray_train/client_env --python 3.12`. Review the
lock diff and run the client dependency tests plus a generation-to-training
contract smoke before deploying. Do not regenerate this lock on TPU startup.
