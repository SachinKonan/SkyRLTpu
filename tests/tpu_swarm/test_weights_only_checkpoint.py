import io
import json
import tarfile
import pytest
from tpu.science.weights_only_checkpoint import strip_optimizer


def test_keeps_exact_adapter_and_drops_adam(tmp_path):
    src, dst = tmp_path/'source.tar.gz', tmp_path/'init.tar.gz'
    payload = {'lora_weights.npz': b'exact trained adapter bytes',
               'optimizer_state.npz': b'old moments and step',
               'tunix_checkpoint_meta.json': json.dumps(dict(
                   format='tunix_backend_v2', lora_config={'rank':32},
                   lora_layouts={'a':'replicated'}, optimizer_layouts={'step':'replicated'})).encode()}
    with tarfile.open(src,'w:gz') as archive:
        for name, data in payload.items():
            info=tarfile.TarInfo(name);info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
    original = src.read_bytes()
    proof = strip_optimizer(src,dst)
    assert src.read_bytes() == original
    assert proof['optimizer_present'] is False
    with tarfile.open(dst) as archive:
        assert archive.extractfile('lora_weights.npz').read() == payload['lora_weights.npz']
        assert 'optimizer_state.npz' not in archive.getnames()
        meta=json.load(archive.extractfile('tunix_checkpoint_meta.json'))
        assert meta['lora_layouts']=={'a':'replicated'} and 'optimizer_layouts' not in meta
    with pytest.raises(ValueError):strip_optimizer(src,src)
    with pytest.raises(ValueError):strip_optimizer(src,dst)


def make_repeated_checkpoint(path, divergent=False):
    import numpy as np
    key = "['adapter']['base']['self_attention']['key']['kernel_lora_b'].value"
    native = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    repeated = np.repeat(native[:, :, None, :], 4, axis=2).reshape(2, 24)
    if divergent:
        repeated[0, 3] += 1
    flat = {key: repeated, 'other': np.arange(5, dtype=np.float32)}
    stream = io.BytesIO()
    np.savez(stream, **flat)
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in {'lora_weights.npz': stream.getvalue(),
            'tunix_checkpoint_meta.json': json.dumps(dict(format='tunix_backend_v2',
                lora_config={'rank': 2}, lora_layouts={key: {'spec': ['tensor']}})).encode()}.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return key, native.reshape(2, 6), flat


def test_native_kv_conversion_is_lossless_and_uses_target_layout(tmp_path):
    import numpy as np
    from tpu.science.weights_only_checkpoint import collapse_repeated_kv
    source, target = tmp_path/'source.tar.gz', tmp_path/'native.tar.gz'
    key, native, original = make_repeated_checkpoint(source)
    before = source.read_bytes()
    proof = collapse_repeated_kv(source, target, native_heads=2, logical_heads=8, head_dim=3)
    assert source.read_bytes() == before
    assert proof['optimizer_present'] is False
    with tarfile.open(target) as archive:
        assert set(archive.getnames()) == {'lora_weights.npz', 'tunix_checkpoint_meta.json'}
        metadata = json.load(archive.extractfile('tunix_checkpoint_meta.json'))
        assert 'lora_layouts' not in metadata
        assert metadata['lora_config'] == {'rank': 2}
        with np.load(io.BytesIO(archive.extractfile('lora_weights.npz').read())) as flat:
            np.testing.assert_array_equal(flat[key], native)
            np.testing.assert_array_equal(flat['other'], original['other'])
            # A @ B has the same projected values after restoring head replicas.
            a = np.array([[0.25, -2], [1, 3]], dtype=np.float32)
            projected = (a @ flat[key]).reshape(2, 2, 3)
            np.testing.assert_array_equal(np.repeat(projected[:, :, None, :], 4, axis=2).reshape(2, 24), a @ original[key])
    with pytest.raises(ValueError, match='must be new'):
        collapse_repeated_kv(source, target, native_heads=2, logical_heads=8, head_dim=3)


def test_divergent_heads_rejected_without_output(tmp_path):
    from tpu.science.weights_only_checkpoint import collapse_repeated_kv
    source, target = tmp_path/'source.tar.gz', tmp_path/'native.tar.gz'
    make_repeated_checkpoint(source, divergent=True)
    before = source.read_bytes()
    with pytest.raises(ValueError, match='replicas diverged'):
        collapse_repeated_kv(source, target, native_heads=2, logical_heads=8, head_dim=3)
    assert source.read_bytes() == before
    assert not target.exists() and not target.with_suffix('.gz.partial').exists()
