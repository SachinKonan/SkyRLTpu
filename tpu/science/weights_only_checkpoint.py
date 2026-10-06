"""Produce a distinct Tunix initialization archive without Adam state.

The current load_weights backend restores optimizer_state.npz whenever present,
regardless of the SDK constructor used. Never modify the original checkpoint.
"""
import hashlib
import io
import json
from pathlib import Path
import tarfile


def collapse_repeated_kv(source, destination, *, native_heads, logical_heads, head_dim):
    """Losslessly convert a weights-only adapter to its native KV-head layout.

    Never average replicas: divergent trained heads cannot be represented by
    this conversion. Saved mesh placements are omitted so the destination
    trainer uses its own template. The original archive is never modified.
    """
    import numpy as np
    import re

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('initialization destination must be new')
    if min(native_heads, logical_heads, head_dim) <= 0 or logical_heads % native_heads:
        raise ValueError('invalid KV head geometry')
    if logical_heads <= native_heads:
        raise ValueError('expected a repeated KV-head source')
    partial = destination.with_suffix(destination.suffix + '.partial')
    changed = []
    try:
        with tarfile.open(source, 'r:gz') as old:
            if set(old.getnames()) != {'lora_weights.npz', 'tunix_checkpoint_meta.json'}:
                raise ValueError('expected weights-only archive; strip optimizer first')
            metadata = json.load(old.extractfile('tunix_checkpoint_meta.json'))
            if metadata.get('format') != 'tunix_backend_v2' or 'lora_mix' in metadata:
                raise ValueError('expected a plain Tunix v2 adapter')
            with np.load(io.BytesIO(old.extractfile('lora_weights.npz').read()), allow_pickle=False) as weights:
                flat = {key: weights[key] for key in weights.files}
        for key, value in flat.items():
            if not re.search(r"\['(?:key|value)'\]\['kernel_lora_b'\]", key):
                continue
            if value.shape[-1] != logical_heads * head_dim:
                raise ValueError(f'Unexpected KV shape for {key}: {value.shape}')
            grouped = value.reshape(*value.shape[:-1], native_heads, logical_heads // native_heads, head_dim)
            canonical = grouped[..., :1, :]
            if not np.all(grouped == canonical):
                raise ValueError(f'K/V LoRA replicas diverged: {key}')
            flat[key] = np.ascontiguousarray(canonical.reshape(*value.shape[:-1], native_heads * head_dim))
            # Verify expansion exactly reconstructs the saved trained tensor.
            expanded = np.repeat(flat[key].reshape(*value.shape[:-1], native_heads, 1, head_dim),
                                 logical_heads // native_heads, axis=-2).reshape(value.shape)
            if not np.array_equal(expanded, value):
                raise ValueError(f'KV roundtrip failed: {key}')
            changed.append(dict(key=key, before=list(value.shape), after=list(flat[key].shape)))
        if not changed:
            raise ValueError('no KV LoRA B tensors found')
        metadata.pop('lora_layouts', None)
        metadata.pop('optimizer_layouts', None)
        encoded = io.BytesIO()
        np.savez(encoded, **flat)
        payloads = {'lora_weights.npz': encoded.getvalue(),
                    'tunix_checkpoint_meta.json': json.dumps(metadata, sort_keys=True).encode()}
        with tarfile.open(partial, 'w:gz', compresslevel=0) as new:
            for name, payload in payloads.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                new.addfile(info, io.BytesIO(payload))
        partial.replace(destination)
        return dict(changed=changed, optimizer_present=False, destination_template_layout=True,
                    lora_config=metadata['lora_config'], lora_sha256=tensor_hash(destination),
                    bytes=destination.stat().st_size)
    finally:
        partial.unlink(missing_ok=True)


def tensor_hash(path):
    with tarfile.open(path, 'r:gz') as archive:
        with archive.extractfile('lora_weights.npz') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
            return digest.hexdigest()


def strip_optimizer(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('initialization destination must be new')
    partial = destination.with_suffix(destination.suffix + '.partial')
    try:
        with tarfile.open(source, 'r:gz') as old:
            members = {m.name: m for m in old if m.isfile()}
            expected = {'lora_weights.npz', 'optimizer_state.npz', 'tunix_checkpoint_meta.json'}
            if set(members) != expected:
                raise ValueError(f'unsupported checkpoint members: {sorted(members)}')
            metadata = json.load(old.extractfile('tunix_checkpoint_meta.json'))
            if metadata.get('format') != 'tunix_backend_v2' or 'lora_mix' in metadata:
                raise ValueError('expected a plain Tunix v2 adapter')
            metadata.pop('optimizer_layouts', None)
            encoded = json.dumps(metadata, sort_keys=True).encode()
            with tarfile.open(partial, 'w:gz', compresslevel=0) as new:
                new.addfile(members['lora_weights.npz'], old.extractfile('lora_weights.npz'))
                info = tarfile.TarInfo('tunix_checkpoint_meta.json')
                info.size = len(encoded)
                new.addfile(info, io.BytesIO(encoded))
        before, after = tensor_hash(source), tensor_hash(partial)
        if before != after:
            raise ValueError('adapter tensor bytes changed')
        with tarfile.open(partial) as archive:
            if set(archive.getnames()) != {'lora_weights.npz', 'tunix_checkpoint_meta.json'}:
                raise ValueError('unexpected weights-only payload')
        partial.replace(destination)
        return dict(lora_sha256=after, optimizer_present=False,
                    lora_config=metadata['lora_config'], bytes=destination.stat().st_size)
    finally:
        partial.unlink(missing_ok=True)
