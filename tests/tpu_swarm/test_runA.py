"""Run A snapshot ensemble arms (muse-s3, muse-s2 on v5p-16; qwen-s3 as an OD v4-64 pool job): profiles (base
settings + hardware + runA patches), v5p Muse farms, K/V conversion / inspection, dry-run setup."""
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tpu import cosearch_collapse_kv as ckv
from tpu import runA_profiles as rp
from tpu import runA_setup as setup
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.regional_preflight import check_paths

ROOT = Path(__file__).resolve().parents[2]
KEY = "['adapter']['base']['decoder']['layers']['layer_3']['attention']['attention']['value']['kernel_lora_b'].value"
MLP = "['adapter']['base']['decoder']['layers']['layer_0']['mlp']['wi_0']['kernel_lora_b'].value"
PATCHED = {'NUM_EPOCHS', 'TTD_LOSS_FN', 'SKYRL_SNAPSHOT_KEEP', 'TTD_IS_CAP'}
RUN_KEYS = {'SKYRL_RUN_ID', 'TTD_SICK_MARKER'}


def _have(ref):
    return subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', ref], capture_output=True).returncode == 0


needs_refs = pytest.mark.skipif(
    not (_have(rp.REF) and _have(rp.MUSE_HARDWARE[0]) and rp.PATCHES.exists()),
    reason='base/co-search commits or rq4-local runA_patches.py not available')


def _patched(ens):
    return PATCHED | {k for k in ens['client_env'] if k.startswith('TTD_SNAPSHOT_')}


def _check_ensemble(plain, ens, arm):
    e = ens['client_env']
    assert (e['NUM_EPOCHS'], e['TTD_LOSS_FN'], e['TTD_IS_CAP'], e['TTD_SNAPSHOT_LAGS'], e['SKYRL_SNAPSHOT_KEEP']) == \
        ('26', 'cispo', '2', '0,2,4,8', '12')
    assert e['TTD_SNAPSHOT_REGISTER'] == ','.join(
        f'{arm}-step{s:02d}:{s}:~/.cache/{rp.RUN_IDS[arm]}/snapshots-src/{arm}-step{s:02d}.tar' for s in (9, 7, 3))
    assert {k: v for k, v in ens.items() if k != 'client_env'} == {k: v for k, v in plain.items() if k != 'client_env'}
    assert {k: v for k, v in e.items() if k not in _patched(ens)} == \
        {k: v for k, v in plain['client_env'].items() if k not in _patched(ens)}
    # CISPO needs the backward warm-up off (the live muse-s3 / qwen-s2 arms run that way).
    assert plain['trainer']['backward_warmup'] is False and ens['inference']['external_pool_target_leases'] == 4


@needs_refs
@pytest.mark.parametrize('arm', ['muse-s3', 'muse-s2'])
def test_muse_trainer_keeps_base_run_settings_on_v5p_hardware(arm):
    out = rp.profiles((arm,), farms=False)
    settings, hw, seed_base = rp.git_profile(*rp.MUSE_SETTINGS), rp.git_profile(*rp.MUSE_HARDWARE), \
        rp.git_profile(*rp.MUSE_BASE[arm])
    run_id = rp.RUN_IDS[arm]
    plain, ens = out[run_id], out[f'{run_id}.ensemble']
    for p in (plain, ens):
        config = Config.from_dict(p)
        check_paths(config.to_dict(), 'us-east5')
        for key in ('model_preset', 'bucket', 'base_bundle', 'base_bundle_sha256', 'zone', 'client_learning_rate',
                    'client_context_window', 'client_phase1_max_tokens', 'grading', 'trainer_env',
                    'single_bucket', 'checkpoint_resume', 'systemd_runtime'):
            assert p[key] == settings[key], key
        # Science settings of both seeds' base runs agree (seed-2 trained on the OD v4-64, seed 3 on v6e-128).
        for key in ('model_preset', 'client_learning_rate', 'client_context_window', 'client_phase1_max_tokens'):
            assert seed_base[key] == settings[key], key
        assert p['inference'] == dict(settings['inference'], external_pool_target_leases=4)
        assert p['inference']['external_pool_discovery_group'] == 'rebut-muse' and p['inference']['remote_only']
        assert (p['accelerator'], p['hosts']) == (hw['accelerator'], hw['hosts'])
        assert p['trainer'] == dict(hw['trainer'], backward_warmup=False)
        assert p['cache'] == dict(hw['cache'], inference_compile=p['cache']['inference_compile'])
        assert (config.trainer.tp, config.trainer.fsdp, config.trainer.logical_kv_heads) == (1, 4, 2)
        assert config.trainer.sequence_length == seed_base['trainer']['sequence_length'] == 28672
        assert not set(rp.COSEARCH_KEYS) & set(p) and not getattr(config, 'trainer_only', False)
        assert config.client_member_spec.rsplit(':', 1)[-1] == setup.ARMS[arm].tag == 'muse'
        assert p['assigned_host_ips'] == [rp.HOSTS[arm]] and p['physical_node'] == rp.NODES[rp.HOSTS[arm]]
        assert p['resume_min_checkpoint_step'] == setup.BATCH == 11
        assert config.grading_memory_gib <= config.cache.reserve_gib - 64
        env = p['client_env']
        assert env['SKYRL_RUN_ID'] == p['run_id'] == run_id and run_id in env['TTD_SICK_MARKER']
    same = {k: v for k, v in seed_base['client_env'].items()
            if k not in _patched(ens) | RUN_KEYS | set(rp.MUSE_DROPPED_ENV)}
    assert {k: plain['client_env'][k] for k in same} == same
    assert set(plain['client_env']) == set(same) | RUN_KEYS | {'NUM_EPOCHS', 'TTD_LOSS_FN'}
    assert plain['client_env']['TTD_M0_LORA_SEED'] == {'muse-s3': '3', 'muse-s2': '2'}[arm]
    _check_ensemble(plain, ens, arm)


@needs_refs
def test_muse_s2_is_muse_s3_but_seed_run_and_host():
    out = rp.profiles(('muse-s3', 'muse-s2'), farms=False)
    s3, s2 = out['rebut-snapA-erdos-muse-s3.ensemble'], out['rebut-snapA-erdos-muse-s2.ensemble']
    diff = {line.split(':')[0] for line in rp.differences(s2, s3)}
    assert diff == {'run_id', 'root', 'assigned_host_ips', 'physical_node', 'cache.inference_compile',
                    'client_env.SKYRL_RUN_ID', 'client_env.TTD_SICK_MARKER', 'client_env.TTD_M0_LORA_SEED',
                    'client_env.TTD_SNAPSHOT_REGISTER'}
    assert rp.profiles(('muse-s2',), muse_s2_host='10.202.15.203', farms=False)[
        'rebut-snapA-erdos-muse-s2']['physical_node'] == rp.NODE_DE70


@needs_refs
def test_qwen_s3_is_the_weights_run_on_the_od_pool_box():
    out = rp.profiles(('qwen-s3',), farms=False)
    base, hw, tree = (rp.git_profile(*r) for r in (rp.QWEN_BASE, rp.QWEN_HARDWARE, rp.QWEN_TREE_RUN))
    plain, ens = out['rebut-snapA-erdos-qwen-s3'], out['rebut-snapA-erdos-qwen-s3.ensemble']
    for p in (plain, ens):
        config = Config.from_dict(p)
        check_paths(config.to_dict(), 'us-central2')
        # Pool job on workers 0-5: trainer on the 2x4x2 box {5,3,2,4}, ranks 0/1 idle (rank 0: head + client).
        assert config.od_v4_64_subset and not config.od_v4_64_partition and not config.assigned_host_ips
        assert config.active_ranks == [2, 3, 4, 5] and config.idle_ranks == (0, 1)
        assert not set(rp.QWEN_PARTITION_KEYS) & set(p)
        assert (p['accelerator'], p['hosts'], p['grading']) == (hw['accelerator'], hw['hosts'], hw['grading'])
        assert p['trainer'] == dict(hw['trainer'], backward_warmup=False)
        assert (config.trainer.hosts, config.trainer.tp, config.trainer.fsdp, config.trainer.process_bounds,
                config.trainer.logical_kv_heads) == (4, 8, 2, '1,2,2', 8) == \
            (4, 8, 2, '1,2,2', setup.ARMS['qwen-s3'].logical_kv_heads)
        assert base['trainer']['logical_kv_heads'] == 8      # what the weights run saved
        for key in ('trainer_gib', 'inference_gib', 'reserve_gib', 'process_count', 'thread_count', 'hf', 'orbax'):
            assert p['cache'][key] == hw['cache'][key], key
        assert p['cache']['trainer_compile_seed'] == hw['cache']['trainer_compile']
        assert p['cache']['trainer_compile'].startswith(f'{rp.CENTRAL2}/rebut/rebut-snapA-erdos-qwen-s3/')
        for key in ('model_preset', 'bucket', 'base_bundle', 'base_bundle_sha256', 'zone', 'client_learning_rate',
                    'client_context_window', 'client_phase1_max_tokens', 'trainer_env', 'checkpoint_resume',
                    'systemd_runtime', 'single_bucket', 'max_restarts_on_errors', 'checkpoint_cleanup_timeout'):
            assert p[key] == base[key], key
        assert p['inference'] == dict(base['inference'], external_pool_discovery_group='rebut-qwen35',
                                      external_pool_target_leases=4)
        assert p['inference']['external_pool_discovery_group'] == tree['inference']['external_pool_discovery_group']
        assert p['resume_min_checkpoint_step'] == 11 and p['bucket'] == setup.ARMS['qwen-s3'].dest_bucket
        env = p['client_env']
        for key in rp.QWEN_REBUTTAL_ENV:
            assert env[key] == tree['client_env'][key], key
        assert env['TTD_M0_LORA_SEED'] == base['client_env']['TTD_M0_LORA_SEED'] == '2'
    same = {k: v for k, v in base['client_env'].items() if k not in _patched(ens) | RUN_KEYS | set(rp.QWEN_REBUTTAL_ENV)}
    assert {k: plain['client_env'][k] for k in same} == same
    _check_ensemble(plain, ens, 'qwen-s3')


@needs_refs
def test_farms_are_the_v6e_muse_farm_on_v5p16_hosts():
    out = rp.profiles(('muse-s3',))
    w05 = rp.git_profile(*rp.FARM_SOURCE)
    trainer = Config.from_dict(out[rp.RUN_IDS['muse-s3']])
    for run_id, host in rp.FARMS.items():
        p = out[run_id]
        config = Config.from_dict(p)
        check_paths(config.to_dict(), 'us-east5')
        assert config.inference_only and config.hosts == 1 and config.engines_per_host == 1
        assert (p['accelerator'], p['assigned_host_ips'], p['physical_node']) == ('tpu-v5p-16', [host], rp.NODES[host])
        assert p['inference'] == w05['inference'] and p['inference']['tp'] == 4
        assert p['cache']['inference_compile'] == f'{rp.EAST5}/rebut/muse-v5p-tp4/inference-compile'
        assert p['cache']['reserve_gib'] == 280 and p['cache']['inference_gib'] + 280 <= 440 - 64
        rest = {k for k in w05 if k not in ('run_id', 'root', 'accelerator', 'assigned_host_ips', 'physical_node',
                                            'cache', 'grading')}
        assert {k: p[k] for k in rest} == {k: w05[k] for k in rest}
        for key in ('max_model_length', 'native_thinking_budget', 'model_impl', 'max_lora_rank', 'serving_runtime',
                    'transformers_version'):
            assert getattr(config.inference, key) == getattr(trainer.inference, key), key


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
    assert ckv.NATIVE[setup.ARMS['muse-s2'].convert] == ckv.NATIVE[setup.ARMS['muse-s3'].convert] == (2, 128)


def test_inspect_reports_the_saved_kv_layout_and_writes_nothing(tmp_path):
    _archive(tmp_path / 'src.tar.gz', logical=8, native=4)
    report = ckv.inspect(tmp_path / 'src.tar.gz', native=4, head_dim=4)
    assert report['logical_kv_heads'] == [8] and report['optimizer_present'] and len(report['kv']) == 3
    assert report['optimizer_counters']["['opt_state']['count'].value"] == 11
    assert sorted(p.name for p in tmp_path.iterdir()) == ['src.tar.gz']
    assert ckv.NATIVE[setup.ARMS['qwen-s3'].inspect_model] == (4, 256) and setup.ARMS['qwen-s3'].convert is None


def test_metrics_keep_batches_before_the_fork():
    rows = ''.join(json.dumps({'progress/batch': b, 'x': b}) + '\n' for b in range(37))
    kept = [json.loads(l)['progress/batch'] for l in setup.kept_metrics(rows).splitlines()]
    assert kept == list(range(11))
    with pytest.raises(SystemExit):
        setup.kept_metrics(''.join(json.dumps({'progress/batch': b}) + '\n' for b in (0, 1, 12)))


def test_tree_best_must_be_the_batch_10_metrics_row():
    tree = {'step': 11, 'states': [{'value': -0.5}, {'value': -0.38}, {'value': None}]}
    rows = ''.join(json.dumps({'progress/batch': b, 'pool/best_value': -0.38 if b >= 10 else -0.4}) + '\n'
                   for b in range(12))
    assert setup.tree_matches_metrics(tree, rows) == -0.38
    with pytest.raises(SystemExit):
        setup.tree_matches_metrics(dict(tree, states=[{'value': -0.37}]), rows)


@pytest.mark.parametrize('arm', sorted(setup.ARMS))
def test_setup_dry_run_runs_nothing(arm, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail(f'dry run executed {a}'))
    assert setup.main(['--arm', arm, '--stage', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    a = setup.ARMS[arm]
    writes = [l for l in out.splitlines() if l.startswith('  gcloud storage cp --if-generation-match=0')]
    dest = f'{a.dest_bucket}/ray-training/{a.dest_run}/'
    # checkpoint + registry + 5 client-log files, all create-only into the new run
    assert len(writes) == 7 and all(l.split()[-1].startswith(dest) for l in writes)
    assert f'{dest}checkpoints/{a.model}/{a.new_ckpt}.tar.gz' in out
    assert f'{a.source_run}/checkpoints/{a.model}/000011.tar.gz' in out
    assert 'puct_sampler_step_000011.json' in out and 'minimal_registry' in out and 'not run' in out
    assert not any(tmp_path.iterdir())
    if arm == 'qwen-s3':
        # unchanged archive, server-side copy inside us-central2; seed 1's lottery tree from us-east5
        assert writes[0].split()[-2] == f'{setup.CENTRAL2}/ray-training/{a.source_run}/checkpoints/{a.model}/000011.tar.gz'
        assert a.new_ckpt == '000011' and '--inspect --model qwen3.5-27b' in out
        assert f'{setup.EAST5}/ray-training/rebut-erdos-qwen35-seed1-splice/client/tinker_log/' \
               'rebut-erdos-qwen35-seed1-splice/puct_sampler_step_000011.json' in out
        assert dest.startswith(setup.CENTRAL2)
    else:
        assert a.new_ckpt == 'v5p-000011' and '--model muse-glimmer-30b --report' in out
        assert dest.startswith(setup.EAST5)
