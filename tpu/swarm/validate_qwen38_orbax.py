"""Validate converted text weights against the pinned HF tensor headers."""
import json
import math
from pathlib import Path
import struct
import sys


def validate(output, snapshot, revision):
    import jax
    import orbax.checkpoint as ocp

    output, snapshot = Path(output), Path(snapshot)
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    expected = 0
    for shard in sorted(set(index['weight_map'].values())):
        with (snapshot / shard).open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            if header_size > 16 * 1024 * 1024:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(header_size))
        for name, tensor in header.items():
            if name.startswith('model.language_model.') or name == 'lm_head.weight':
                expected += math.prod(tensor['shape'])
    items = output / '0/items'
    if not (items / 'manifest.ocdbt').is_file():
        raise ValueError('Missing committed checkpoint manifest')
    metadata = ocp.PyTreeCheckpointer().metadata(str(items))
    tree = metadata.item_metadata.tree
    arrays = [value for value in jax.tree_util.tree_leaves(tree)
              if hasattr(value, 'shape') and value.shape]
    actual = sum(math.prod(value.shape) for value in arrays)
    if actual != expected or expected != 26_895_998_464:
        raise ValueError(f'Checkpoint parameter mismatch: Orbax={actual}, HF text={expected}')
    if any(str(value.dtype) != 'bfloat16' for value in arrays):
        raise ValueError('Unexpected converted weight dtype')
    receipt = dict(hf_model='Qwen/Qwen3.8-27B', hf_revision=revision,
                   architecture='qwen3.5-27b', parameter_count=actual, arrays=len(arrays),
                   checkpoint_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                   maxtext_commit='0fd409939977ac0ab79a4e64d21730936f253567')
    temporary = output / 'CHECKPOINT_COMPLETE.tmp'
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(output / 'CHECKPOINT_COMPLETE')
    return receipt


if __name__ == '__main__':
    print(json.dumps(validate(*sys.argv[1:])), flush=True)
