"""Compare loaded model snapshots and installed serving code, not model aliases.

The compatibility hash covers only what changes what is generated for a request: model, weights and tokenizer
files, chat template and generation config, serving and thinking-budget code, serving packages, model
implementation, context length and LoRA rank. Farms of one model on different hardware layouts (v5p TP2,
v6e TP8) therefore share it and are interchangeable. The hardware/runtime layout (accelerator, tensor
parallelism, engines, batch size, memory use, sliding-window KV, engine env) is reported separately as
``runtime``. It is stamped on every completion (``served_by``) so sampler differences between layouts can be
corrected (importance sampling) instead of locking a run to one layout."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(config, root, source, snapshot):
    root, source, snapshot = map(Path, (root, source, snapshot))
    files = {}
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'generation_config.json', 'model.safetensors.index.json'):
        path = snapshot / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not {'config.json', 'tokenizer.json'} <= files.keys():
        raise RuntimeError('serving identity requires model and tokenizer metadata')
    code = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ('tpu/vllm_tpu_server.py', 'tpu/thinking_budget/server.py')
            + (('tpu/vllm_tpu_server_v032.py',) if config.inference.serving_runtime == 'v0.32' else ())}
    packages = {}
    for site in (root / 'envs' / config.inference.serving_env / 'lib').glob('python*/site-packages'):
        for dist in importlib.metadata.distributions(path=[str(site)]):
            name = dist.metadata['Name'].lower().replace('_', '-')
            if name in ('vllm', 'vllm-tpu', 'tpu-inference', 'jax', 'jaxlib', 'torch', 'torchax', 'transformers'):
                provenance = json.loads(dist.read_text('direct_url.json') or '{}')
                # Installation directories differ by run; compare content and
                # immutable VCS/archive provenance rather than local file URLs.
                source_hashes = {}
                for file in dist.files or []:
                    if str(file).endswith('.py'):
                        path = dist.locate_file(file)
                        if path.is_file():
                            source_hashes[str(file)] = hashlib.sha256(path.read_bytes()).hexdigest()
                if not source_hashes or provenance.get('dir_info', {}).get('editable'):
                    raise RuntimeError(f'{name} needs a materialized serving install for attestation')
                packages[name] = dict(version=dist.version,
                    commit=provenance.get('vcs_info', {}).get('commit_id'),
                    archive=provenance.get('archive_info', {}).get('hashes'),
                    python_sha256=digest(source_hashes))
    if (not {'tpu-inference', 'transformers', 'jax', 'jaxlib'} <= packages.keys()
            or not {'vllm', 'vllm-tpu'} & packages.keys()):
        raise RuntimeError('installed serving package identity is incomplete')
    contract = dict(model=config.model, revision=snapshot.name, files=files, code=code,
                    packages=packages, native_thinking=config.inference.native_thinking_budget,
                    model_impl=config.inference.model_impl,
                    max_model_length=config.inference.max_model_length,
                    max_lora_rank=config.inference.max_lora_rank)
    # Layout and performance settings: they change numerics (reduction order), not what a request asks for.
    # Sliding-window KV is on by default in the installed tpu-inference; SKYRL_SWA_KV=0 turns it off.
    runtime = dict(accelerator=config.accelerator, tensor_parallel=config.inference.tp,
                   engines_per_host=config.engines_per_host, max_sequences=config.inference.max_sequences,
                   memory_utilization=config.inference.memory_utilization,
                   sliding_window_kv=(config.inference.engine_env or {}).get('SKYRL_SWA_KV', '1') != '0',
                   engine_env_sha256=digest(config.inference.engine_env or {}))  # values may hold credentials
    return dict(contract=contract, runtime=runtime, sha256=digest(contract), runtime_sha256=digest(runtime))
