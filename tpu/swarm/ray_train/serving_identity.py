"""Compare loaded model snapshots and installed serving code, not model aliases."""
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
            for name in ('tpu/vllm_tpu_server.py', 'tpu/thinking_budget/server.py')}
    packages = {}
    for site in (root / 'envs/serving/lib').glob('python*/site-packages'):
        for dist in importlib.metadata.distributions(path=[str(site)]):
            name = dist.metadata['Name'].lower().replace('_', '-')
            if name in ('vllm', 'tpu-inference', 'jax', 'jaxlib', 'torch', 'torchax', 'transformers'):
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
    if not {'vllm', 'tpu-inference', 'transformers', 'jax', 'jaxlib'} <= packages.keys():
        raise RuntimeError('installed serving package identity is incomplete')
    contract = dict(model=config.model, revision=snapshot.name, files=files, code=code,
                    packages=packages, native_thinking=config.inference.native_thinking_budget,
                    model_impl=config.inference.model_impl,
                    max_model_length=config.inference.max_model_length,
                    max_lora_rank=config.inference.max_lora_rank)
    return dict(contract=contract, sha256=digest(contract))
