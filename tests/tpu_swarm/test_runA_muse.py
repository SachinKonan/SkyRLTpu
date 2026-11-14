"""Run A snapshot ensemble, Muse seed 3 on v5p-16: profiles (base settings + v5p hardware + runA patches), v5p Muse
farms, K/V conversion with Adam, dry-run setup."""
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tpu import cosearch_collapse_kv as ckv
from tpu import runA_muse_profiles as rp
from tpu import runA_muse_setup as setup
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.regional_preflight import check_paths

ROOT = Path(__file__).resolve().parents[2]
KEY = "['adapter']['base']['decoder']['layers']['layer_3']['attention']['attention']['value']['kernel_lora_b'].value"
MLP = "['adapter']['base']['decoder']['layers']['layer_0']['mlp']['wi_0']['kernel_lora_b'].value"


def _have(ref):
    return subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', ref], capture_output=True).returncode == 0


needs_refs = pytest.mark.skipif(
    not (_have(rp.BASE[0]) and _have(rp.HARDWARE[0]) and rp.PATCHES.exists()),
    reason='base/co-search commits or rq4-local runA_patches.py not available')


@needs_refs
def test_trainer_profile_keeps_base_run_settings_on_v5p_hardware():
    out = rp.profiles()
    base, hw = rp.git_profile(*rp.BASE), rp.git_profile(*rp.HARDWARE)
    plain, ens = out[rp.RUN_ID], out[f'{rp.RUN_ID}.ensemble']
    for p in (plain, ens):
        config = Config.from_dict(p)
        check_paths(config.to_dict(), 'us-east5')
        # Base run: what a candidate sees, how it is scored, how it samples and trains (except Run A's patches).
        for key in ('model_preset', 'bucket', 'base_bundle', 'base_bundle_sha256', 'zone', 'client_learning_rate',
                    'client_context_window', 'client_phase1_max_tokens', 'grading', 'trainer_env',
                    'single_bucket', 'checkpoint_resume', 'systemd_runtime'):
            assert p[key] == base[key], key
        assert p['inference'] == dict(base['inference'], external_pool_target_leases=4)
        assert p['inference']['external_pool_discovery_group'] == 'rebut-muse' and p['inference']['remote_only']
        # v5p-16 trainer block, RAM sizing and compile cache of the co-search Muse trainer.
        assert (p['accelerator'], p['hosts'], p['trainer']) == (hw['accelerator'], hw['hosts'], hw['trainer'])
        assert p['cache'] == dict(hw['cache'], inference_compile=p['cache']['inference_compile'])
        assert (config.trainer.tp, config.trainer.fsdp, config.trainer.logical_kv_heads) == (1, 4, 2)
        assert config.trainer.sequence_length == base['trainer']['sequence_length'] == 28672
        assert config.trainer.seq_buckets == base['trainer']['seq_buckets']
        # A normal single run (client + trainer), not the co-search trainer-only host.
        assert not set(rp.COSEARCH_KEYS) & set(p) and not getattr(config, 'trainer_only', False)
        assert config.client_member_spec.rsplit(':', 1)[-1] == setup.TAG == 'muse'
        assert p['assigned_host_ips'] == ['10.202.15.201'] and p['physical_node'] == rp.NODE_5NMB
        assert p['resume_min_checkpoint_step'] == setup.BATCH == 11
        assert config.grading_memory_gib <= config.cache.reserve_gib - 64
        env = p['client_env']
        assert env['SKYRL_RUN_ID'] == p['run_id'] == rp.RUN_ID and rp.RUN_ID in env['TTD_SICK_MARKER']
        assert (env['SKYRL_OPEN_GRADING'], env['SKYRL_GRADING_REGISTRY'], env['SKYRL_OPEN_GRADING_K']) == \
            ('1', '/home/gcpuser/.cache/skyrl/rebut-farms.json', '20')
    patched = {'NUM_EPOCHS', 'TTD_LOSS_FN', 'SKYRL_SNAPSHOT_KEEP', 'TTD_IS_CAP'} | {
        k for k in ens['client_env'] if k.startswith('TTD_SNAPSHOT_')}
    same = {k: v for k, v in base['client_env'].items() if k not in patched | {'SKYRL_RUN_ID', 'TTD_SICK_MARKER'}}
    assert {k: plain['client_env'][k] for k in same} == same == {k: ens['client_env'][k] for k in same}
    assert plain['client_env']['NUM_EPOCHS'] == '37' and plain['client_env']['TTD_LOSS_FN'] == 'importance_sampling'
    e = ens['client_env']
    assert (e['NUM_EPOCHS'], e['TTD_LOSS_FN'], e['TTD_IS_CAP'], e['TTD_SNAPSHOT_LAGS'], e['SKYRL_SNAPSHOT_KEEP']) == \
        ('26', 'cispo', '2', '0,2,4,8', '12')
    assert e['TTD_SNAPSHOT_REGISTER'] == ','.join(
        f'muse-s3-step{s:02d}:{s}:~/.cache/{rp.RUN_ID}/snapshots-src/muse-s3-step{s:02d}.tar' for s in (9, 7, 3))
    assert {k: v for k, v in ens.items() if k != 'client_env'} == {k: v for k, v in plain.items() if k != 'client_env'}


@needs_refs
def test_farms_are_the_v6e_muse_farm_on_v5p16_hosts():
    out = rp.profiles()
    w05 = rp.git_profile(*rp.FARM_SOURCE)
    trainer = Config.from_dict(out[rp.RUN_ID])
    assert sorted(rp.FARMS.values()) == ['10.202.15.200', '10.202.15.203', '10.202.15.204']
    for run_id, host in rp.FARMS.items():
        p = out[run_id]
        config = Config.from_dict(p)
        check_paths(config.to_dict(), 'us-east5')
        assert config.inference_only and config.hosts == 1 and config.engines_per_host == 1
        assert (p['accelerator'], p['assigned_host_ips'], p['physical_node']) == ('tpu-v5p-16', [host], rp.NODES[host])
        assert p['inference'] == w05['inference'] and p['inference']['tp'] == 4
        assert p['inference']['batched_rpa_kernel'] is False and p['inference']['prefix_caching'] is True
        assert p['cache']['inference_compile'] == f'{rp.BUCKET}/rebut/muse-v5p-tp4/inference-compile'
        assert p['cache']['inference_compile'] != w05['cache']['inference_compile']
        assert p['cache']['reserve_gib'] == 280 and p['cache']['inference_gib'] + 280 <= 440 - 64
        assert config.grading_families['math']['slots_per_host'] == 48
        assert config.grading_families['placement']['slots_per_host'] == 48
        rest = {k for k in w05 if k not in ('run_id', 'root', 'accelerator', 'assigned_host_ips', 'physical_node',
                                            'cache', 'grading')}
        assert {k: p[k] for k in rest} == {k: w05[k] for k in rest}
        # Compatibility-hash inputs match the trainer's farms (accelerator and TP are runtime, not contract).
        for key in ('model', 'max_model_length', 'native_thinking_budget', 'model_impl', 'max_lora_rank',
                    'serving_runtime', 'transformers_version'):
            a = getattr(config, key, None) if key == 'model' else getattr(config.inference, key)
            b = getattr(trainer, key, None) if key == 'model' else getattr(trainer.inference, key)
            assert a == b, key


@needs_refs
def test_written_profiles_are_the_built_ones():
    for name, p in rp.profiles().items():
        assert json.loads((rp.PROFILES / f'{name}.json').read_text()) == p, name


def _archive(path, logical=8, native=2, hd=4):
    rng = np.random.default_rng(1)
    native_b = rng.standard_normal((2, 3, native * hd)).astype(np.float32)
    repeated = np.repeat(native_b.reshape(2, 3, native, 1, hd), logical // native, axis=-2).reshape(2, 3, logical * hd)
    mlp = rng.standard_normal((2, 3, 5)).astype(np.float32)
    opt = {f"['opt_state']['inner_state'][0]['mu']{KEY}": repeated * 0.25,
           f"['opt_state']['inner_state'][0]['nu']{KEY}": repeated ** 2,
           f"['opt_state']['inner_state'][0]['mu']{MLP}": mlp, "['opt_state']['count'].value": np.array(11, np.int32)}
    meta = dict(format='tunix_backend_v2', lora_config={'rank': 32}, lora_layouts={KEY: {}}, optimizer_layouts={})
    with tarfile.open(path, 'w:gz') as t:
        for name, data in (('lora_weights.npz', {KEY: repeated, MLP: mlp}), ('optimizer_state.npz', opt)):
            buf = io.BytesIO()
            np.savez(buf, **data)
            info = tarfile.TarInfo(name)
            info.size = len(buf.getvalue())
            t.addfile(info, io.BytesIO(buf.getvalue()))
        raw = json.dumps(meta).encode()
        info = tarfile.TarInfo('tunix_checkpoint_meta.json')
        info.size = len(raw)
        t.addfile(info, io.BytesIO(raw))
    return native_b


def test_muse_kv_collapse_keeps_adam(tmp_path):
    native_b = _archive(tmp_path / 'src.tar.gz')
    report = ckv.convert(tmp_path / 'src.tar.gz', tmp_path / 'dst.tar.gz', native=2, head_dim=4)
    with tarfile.open(tmp_path / 'dst.tar.gz', 'r:gz') as t:
        files = {m.name: t.extractfile(m).read() for m in t if m.isfile()}
    opt = dict(np.load(io.BytesIO(files['optimizer_state.npz'])))
    assert np.array_equal(dict(np.load(io.BytesIO(files['lora_weights.npz'])))[KEY], native_b)
    assert np.array_equal(opt[f"['opt_state']['inner_state'][0]['nu']{KEY}"], native_b ** 2)
    assert report['optimizer_counters']["['opt_state']['count'].value"] == 11
    assert all(v['replicas_per_native_head'] == 4 for v in report['converted'].values())
    assert 'lora_layouts' not in json.loads(files['tunix_checkpoint_meta.json'])
    assert ckv.NATIVE[setup.CONVERT] == (2, 128)


def test_metrics_keep_batches_before_the_fork():
    rows = ''.join(json.dumps({'progress/batch': b, 'x': b}) + '\n' for b in range(37))
    kept = [json.loads(l)['progress/batch'] for l in setup.kept_metrics(rows).splitlines()]
    assert kept == list(range(11))
    with pytest.raises(SystemExit):
        setup.kept_metrics(''.join(json.dumps({'progress/batch': b}) + '\n' for b in (0, 1, 12)))


def test_setup_dry_run_runs_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail(f'dry run executed {a}'))
    assert setup.main(['--stage', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    writes = [l for l in out.splitlines() if l.startswith('  gcloud storage cp --if-generation-match=0')]
    dest = f'{setup.RUNS}/{setup.DEST_RUN}/'
    # converted checkpoint + registry + 5 client-log files, all create-only into the new run
    assert len(writes) == 7 and all(l.split()[-1].startswith(dest) for l in writes)
    assert f'{dest}checkpoints/model_92e8ec42/v5p-000011.tar.gz' in out
    assert 'rebut-erdos-muse-seed3/checkpoints/model_92e8ec42/000011.tar.gz' in out
    assert 'puct_sampler_step_000011.json' in out and 'minimal_registry' in out
    assert '--model muse-glimmer-30b' in out and 'not run' in out
    assert not any(tmp_path.iterdir())
