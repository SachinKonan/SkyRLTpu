import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.commands import trainer_environment, inference_environment, client_environment
from tpu.swarm.ray_train.runtime_inventory import clean_url, differences
from tpu.swarm.ray_train.launch_contract import write_launch_contract
from tpu.swarm.ray_train.thinking_budget.contract import check
from tpu.swarm.ray_train.warmup_contract import shapes

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'tpu/swarm/ray_train/profiles/science-placement-v6e-qwen-cpu-seeded-train-001.json'


def raw():
    return json.loads(PROFILE.read_text())


def test_ambient_diagnostic_and_compiler_state_does_not_reach_workloads(monkeypatch):
    for key in ('TUNIX_GRADIENT_PROBE_DIR', 'XLA_FLAGS', 'TTD_INIT_STATE_PATH_QWEN',
                'CUSTOM_NUM_TOKENS_BUCKETS', 'SERIALIZE_MODEL_AND_SAMPLING'):
        monkeypatch.setenv(key, 'inherited-bad-value')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', '/private/credential-path')
    cfg = Config.from_dict(raw())
    envs = [trainer_environment(cfg, Path('/cache'), Path('/run'), ['h0'], 0),
            inference_environment(cfg, Path('/cache'), Path('/run')),
            client_environment(cfg, Path('/cache'), 'h0')]
    for env in envs:
        assert 'TUNIX_GRADIENT_PROBE_DIR' not in env
        assert 'XLA_FLAGS' not in env
        assert 'TTD_INIT_STATE_PATH_QWEN' not in env
        assert 'inherited-bad-value' not in env.values()
        assert env['GOOGLE_APPLICATION_CREDENTIALS'] == '/private/credential-path'


def test_explicit_diagnostics_still_reach_trainer_only():
    data = raw(); data['trainer_env']['TUNIX_REPLAY_DIAGNOSTICS'] = '1'
    cfg = Config.from_dict(data)
    assert trainer_environment(cfg, Path('/cache'), Path('/run'), ['h0'], 0)['TUNIX_REPLAY_DIAGNOSTICS'] == '1'
    assert 'TUNIX_REPLAY_DIAGNOSTICS' not in inference_environment(cfg, Path('/cache'), Path('/run'))


@pytest.mark.parametrize('section,key,value', [
    ('trainer_env', 'TUNIX_ROW_SHARD', '8'),
    ('trainer_env', 'TPU_PROCESS_PORT', '12345'),
    ('trainer_env', 'SKYRL_DATABASE_URL', 'sqlite:///wrong'),
    ('engine_env', 'TPU_VISIBLE_CHIPS', '3'),
    ('engine_env', 'USE_JAX_RAGGED_CONV1D', '0'),
])
def test_conflicting_overrides_fail_before_launch(section, key, value):
    data = raw()
    target = data['trainer_env'] if section == 'trainer_env' else data['inference'].setdefault('engine_env', {})
    target[key] = value
    with pytest.raises(ValueError):
        Config.from_dict(data)


@pytest.mark.parametrize('flag', ['--tensor-parallel-size=2', '--max-num-seqs', '--port=9999'])
def test_duplicate_managed_cli_flags_rejected(flag):
    data = raw(); data['inference']['extra_args'] = [flag]
    with pytest.raises(ValueError, match='managed'):
        Config.from_dict(data)


def test_inventory_comparison_finds_transitive_package_and_source_drift():
    expected = dict(schema=1, python='3.12.0', machine='x86_64', packages={'tokenizers': {'version':'0.22.2'}}, sources={'patch':'hash'})
    actual = copy.deepcopy(expected)
    assert differences(expected, actual) == {}
    actual['packages']['tokenizers']['version'] = '0.23.1'
    actual['sources']['patch'] = 'changed'
    assert set(differences(expected, actual)) == {'packages', 'sources'}
    assert clean_url('https://user:secret@example.com/a?token=secret#secret') == 'https://example.com/a'
    assert clean_url('file:///private/checkout') == 'file:<local-source>'


def test_launch_manifest_excludes_credentials(tmp_path):
    path = tmp_path / 'launch.json'
    write_launch_contract(path, ['python', '-m', 'trainer', '--api-key', 'secret',
                               '--backend-config', '{"url":"https://user:secret@host/path?token=secret"}'], {'GOOGLE_APPLICATION_CREDENTIALS':'secret',
        'TINKER_API_KEY':'secret', 'TTD_AUTH_TOKEN':'secret', 'TUNIX_ROW_SHARD':'2'})
    assert 'secret' not in path.read_text()
    assert json.loads(path.read_text())['environment'] == {'TUNIX_ROW_SHARD':'2'}


def test_native_contract_checks_actual_client_and_rejects_drift(tmp_path):
    assert check(ROOT)
    relative = Path('third_party/discover/ttt_discover/tinker_utils/completers.py')
    destination = tmp_path / relative; destination.parent.mkdir(parents=True)
    destination.write_text((ROOT / relative).read_text().replace('Here is the final complete program:', 'Different cue:'))
    with pytest.raises(ValueError, match='contract differs'):
        check(tmp_path)


def test_native_contract_runs_without_site_packages():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-B', '-S', '-c',
        'from tpu.swarm.ray_train.thinking_budget.contract import check; assert check(".")'],
        cwd=ROOT, check=True, capture_output=True, text=True)


def test_warmup_shapes_match_production_buckets():
    assert shapes(22528, 45056, 2, buckets=[18432,22528]) == [(2,18432),(2,22528)]
    assert shapes(22528, 90112, 4, buckets=[18432,22528]) == [(4,18432),(4,22528)]
    with pytest.raises(ValueError, match='maximum'):
        shapes(22528, 45056, 2, buckets=[18432])


@pytest.mark.parametrize('fail', [False, True])
def test_backward_warmup_preserves_accumulator_and_optimizer(monkeypatch, fail):
    from skyrl.backends.backward_warmup import run
    weights, optimizer, old_grad = object(), object(), object()
    slot = SimpleNamespace(mix=None, lora_state=weights, optimizer=optimizer, accum_grads=old_grad, accum_count=7)
    seen = []
    def backward(batch, *, with_grads):
        assert with_grads
        assert slot.accum_grads is None and slot.accum_count == 0
        seen.append((len(batch.all_model_ids), len(batch.all_targets[0])))
        assert all(not any(row) for row in batch.all_token_weights)
        slot.accum_grads = [0.]; slot.accum_count = len(batch.all_model_ids)
        if fail:
            raise RuntimeError('compiler failed')
        return {}
    backend = SimpleNamespace(models={'m':slot}, config=SimpleNamespace(train_token_budget=16),
                              _row_shard=lambda:2, _model_pass=backward)
    monkeypatch.setenv('TUNIX_UNIFORM_SEQ_LEN','0')
    monkeypatch.setenv('TUNIX_SEQ_BUCKETS','4,8')
    monkeypatch.setenv('TUNIX_WARMUP_MAX_LENGTH','8')
    if fail:
        with pytest.raises(RuntimeError, match='compiler failed'):
            run(backend, 'm')
    else:
        run(backend, 'm')
        assert seen == [(4,4),(2,8)]
    assert slot.accum_grads is old_grad and slot.accum_count == 7
    assert slot.optimizer is optimizer and slot.lora_state is weights


def test_warmup_is_explicit_and_legacy_dummy_is_disabled():
    data=raw(); data['trainer']['backward_warmup']=True
    cfg=Config.from_dict(data)
    assert cfg.requires_source_overlay
    assert client_environment(cfg, Path('/cache'), 'h0')['TTD_WARMUP_FB'] == '0'
    assert trainer_environment(cfg, Path('/cache'), Path('/run'), ['h0'], 0)['TUNIX_BACKWARD_WARMUP'] == '1'
    from tpu.swarm.ray_train.overlay import manifest
    files=manifest(ROOT,cfg)
    assert 'skyrl/backends/backward_warmup.py' in files
    assert 'tpu/swarm/ray_train/warmup_contract.py' in files


@pytest.mark.parametrize('role', ['trainer', 'inference', 'client'])
def test_reviewed_runtime_baseline_blocks_mismatched_environment(tmp_path, monkeypatch, role):
    from tpu.swarm.ray_train import host as module
    folder = tmp_path / 'env'; folder.mkdir()
    run = tmp_path / 'run'; run.mkdir()
    baseline_dir = tmp_path / 'runtime_baselines'; baseline_dir.mkdir()
    expected = dict(schema=1, python='3.12.0', machine='x86_64', packages={'jax': {'version':'old'}}, sources={})
    (baseline_dir / 'approved.json').write_text(json.dumps(expected))
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'host.py'))
    baseline_role = 'serving' if role == 'inference' else role
    def checked(name, command):
        actual = copy.deepcopy(expected); actual['packages']['jax']['version']='changed'
        (run / f'runtime-{baseline_role}-0.json').write_text(json.dumps(actual))
    host = SimpleNamespace(run=run, source=tmp_path, rank=0, checked=checked,
                           config=SimpleNamespace(runtime_baselines={baseline_role:'approved.json'}))
    with pytest.raises(RuntimeError, match='reviewed runtime baseline'):
        module.Host.verify_runtime(host, role, folder, fresh=True)
    assert not (folder / '.runtime-inventory.json').exists()


@pytest.mark.parametrize('flag', ['-tp', '-pp=2', '-dp', '--max_model_len=10',
    '--download_dir=/tmp/wrong', '--enable_lora', '--no-enable-lora',
    '--distributed_executor_backend=mp', '--limit_mm_per_prompt={}'])
def test_cli_aliases_cannot_override_managed_settings(flag):
    data = raw(); data['inference']['extra_args'] = [flag]
    with pytest.raises(ValueError, match='managed'):
        Config.from_dict(data)


@pytest.mark.parametrize('key', ['VLLM_XLA_CACHE_PATH', 'JAX_COMPILATION_CACHE_DIR',
    'VLLM_LORA_RESOLVER_CACHE_DIR', 'VLLM_PLUGINS', 'HF_HUB_OFFLINE'])
def test_engine_environment_cannot_redirect_owned_caches(key):
    data = raw(); data['inference'].setdefault('engine_env', {})[key] = 'wrong'
    with pytest.raises(ValueError, match='executor'):
        Config.from_dict(data)


@pytest.mark.parametrize('enabled', [False, True])
def test_prefix_cache_flag_is_explicit_and_chunked_prefill_preserves_legacy_default(enabled):
    from tpu.swarm.ray_train.commands import inference_command
    data = raw(); data['inference'].update(prefix_caching=enabled, chunked_prefill=enabled)
    cfg = Config.from_dict(data)
    cmd = inference_command(cfg, Path('/cache'), Path('/source'), Path('/model'), Path('/run'))
    assert ('--enable-prefix-caching' in cmd) is enabled
    assert ('--enable-chunked-prefill' in cmd) is enabled
    assert ('--no-enable-prefix-caching' in cmd) is (not enabled)
    assert '--no-enable-chunked-prefill' not in cmd


def test_launch_record_preserves_sqlite_and_records_client_settings(tmp_path):
    env = client_environment(Config.from_dict(raw()), Path('/cache'), 'head')
    env.update(SKYRL_DATABASE_URL='sqlite:////cache/run/tinker.db', PYTHONPATH='/source',
               SOME_UNKNOWN_CREDENTIAL='never-record-this')
    path = tmp_path / 'launch.json'
    write_launch_contract(path, ['python', 'client.py'], env)
    record = json.loads(path.read_text())['environment']
    for key in ('HF_HOME', 'HF_HUB_OFFLINE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'GROUP_SIZE', 'LEARNING_RATE', 'PYTHONPATH', 'SKYRL_DATABASE_URL'):
        assert record[key] == env[key]
    assert 'never-record-this' not in path.read_text()
    assert clean_url('https://user:secret@host/path?token=secret') == 'https://host/path'


def _client_source(tmp_path, transform):
    relative = Path('third_party/discover/ttt_discover/tinker_utils/completers.py')
    target = tmp_path / relative; target.parent.mkdir(parents=True)
    target.write_text(transform((ROOT / relative).read_text()))
    return tmp_path


@pytest.mark.parametrize('change', ['inheritance', 'annotated', 'tuple', 'postclass', 'method', 'missing'])
def test_native_contract_rejects_unreviewed_client_changes(tmp_path, change):
    def transform(text):
        if change == 'inheritance':
            return text.replace('class GemmaTwoPhaseTokenCompleter(QwenTwoPhaseTokenCompleter)',
                                'class GemmaTwoPhaseTokenCompleter(TokenCompleter)')
        if change == 'annotated':
            return text.replace('THINK_CLOSE = "<channel|>"', 'THINK_CLOSE: str = "wrong"')
        if change == 'tuple':
            return text.replace('THINK_CLOSE = "<channel|>"', 'THINK_CLOSE, other = "wrong", 1')
        if change == 'postclass':
            return text + '\nGemmaTwoPhaseTokenCompleter.ANSWER_CUE = "wrong"\n'
        if change == 'method':
            return text.replace('transition = self.tokenizer.encode(self.THINK_CLOSE + self.ANSWER_CUE,',
                                'transition = self.tokenizer.encode(self.THINK_CLOSE,')
        return text.replace('class GemmaTwoPhaseTokenCompleter(', 'class MissingCompleter(')
    source = _client_source(tmp_path, transform)
    with pytest.raises(ValueError, match='contract error'):
        check(source, model='gemma4-31b')


def test_native_checks_only_selected_model(tmp_path):
    source = _client_source(tmp_path, lambda text: text.replace(
        'class MuseTwoPhaseTokenCompleter(', 'class AbsentMuseCompleter('))
    assert check(source, model='qwen3.5-27b')
    with pytest.raises(ValueError, match='contract error'):
        check(source, model='muse-glimmer-30b')


def test_inference_only_does_not_need_discover(tmp_path):
    assert check(tmp_path, model='muse-glimmer-30b', require_client=False)


def test_warmup_matches_oversized_singleton_padding():
    assert shapes(22528, 18000, 4, buckets=[18432, 22528]) == [(4,18432), (4,22528)]


def test_multihost_warmup_never_broadcasts_nested_rpc(monkeypatch):
    from skyrl.backends.tunix_backend import DistributedTunixBackend, TunixBackend
    from skyrl.backends.backward_warmup import run
    calls = []
    def model_pass(self, batch, *, with_grads):
        assert with_grads
        calls.append((self.process_id, len(batch.all_targets[0])))
        self.models['m'].accum_grads = [0.]
        return {}
    def nested_rpc(*args, **kwargs):
        pytest.fail('warmup issued a nested RPC from inside create_model')
    monkeypatch.setattr(TunixBackend, '_model_pass', model_pass)
    monkeypatch.setattr(DistributedTunixBackend, '_broadcast_and_call', nested_rpc)
    monkeypatch.setattr(TunixBackend, '_row_shard', lambda self: 2)
    monkeypatch.setenv('TUNIX_UNIFORM_SEQ_LEN', '0')
    monkeypatch.setenv('TUNIX_SEQ_BUCKETS', '4,8')
    monkeypatch.setenv('TUNIX_WARMUP_MAX_LENGTH', '8')
    for rank in range(4):
        backend = object.__new__(DistributedTunixBackend if rank == 0 else TunixBackend)
        backend.process_id = rank
        backend.config = SimpleNamespace(train_token_budget=16)
        original = object()
        backend.models = {'m': SimpleNamespace(mix=None, accum_grads=original, accum_count=7)}
        run(backend, 'm')
        assert backend.models['m'].accum_grads is original
        assert backend.models['m'].accum_count == 7
    assert calls == [(rank, size) for rank in range(4) for size in (4, 8)]


def test_paired_engine_worker_strips_ambient_and_receives_bucket_flags(monkeypatch):
    # Extract only the production hook: importing the vLLM server requires TPU
    # packages, but this hook itself intentionally has no such dependency.
    import ast
    tree = ast.parse((ROOT / 'tpu/vllm_tpu_server.py').read_text())
    hook = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_sanitize_ray_worker_environment')
    keys = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_WORKER_ENV_KEYS' for t in n.targets))
    namespace = {}
    exec(compile(ast.Module(body=[hook, keys], type_ignores=[]), '<worker-env>', 'exec'), namespace)
    for key in ('XLA_FLAGS', 'TUNIX_GRADIENT_PROBE_DIR', 'TPU_VISIBLE_CHIPS'):
        monkeypatch.setenv(key, 'wrong')
    expected = {'CUSTOM_NUM_TOKENS_BUCKETS': '64,128', 'SERIALIZE_MODEL_AND_SAMPLING': '1',
                'USE_JAX_RAGGED_CONV1D': '0'}
    for key in expected:
        monkeypatch.setenv(key, 'wrong')
        assert key in namespace['_WORKER_ENV_KEYS']
    namespace['_sanitize_ray_worker_environment'](expected)
    import os
    assert all(os.environ[key] == value for key, value in expected.items())
    assert all(key not in os.environ for key in ('XLA_FLAGS', 'TUNIX_GRADIENT_PROBE_DIR', 'TPU_VISIBLE_CHIPS'))


def test_paired_engine_fix_is_in_deployed_overlay():
    from tpu.swarm.ray_train.overlay import manifest
    cfg = Config.load(ROOT / 'tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json')
    assert 'tpu/vllm_tpu_server.py' in manifest(ROOT, cfg)


def test_gemma_tp8_uses_global_kv_heads():
    from tpu.swarm.ray_train.commands import maxtext_kwargs
    cfg = SimpleNamespace(model_preset='gemma4-31b', trainer=SimpleNamespace(
        tp=8, fsdp=2, remat='full', num_vocab_tiling=32, tokamax_splash=True,
        logical_kv_heads=8, maxtext_kwargs={}))
    kwargs = maxtext_kwargs(cfg, Path('/cache'))
    assert kwargs['global_num_kv_heads'] == 8
    assert 'base_num_kv_heads' not in kwargs


def test_client_launch_writes_effective_record(tmp_path):
    from tpu.swarm.ray_train.host import Host
    data = raw(); data['client_env']['CUSTOM_GRADER_MODE'] = 'profile-value'
    cfg = Config.from_dict(data)
    calls = []
    host = SimpleNamespace(rank=0, root=tmp_path, run=tmp_path, source=ROOT, config=cfg,
        ips=['head'], trainer_leader=0, heartbeat=lambda: {},
        start=lambda *args: calls.append(args))
    Host.start_client(host)
    record = json.loads((tmp_path/'launch-client-0.json').read_text())
    assert record['command'] == calls[0][1]
    assert record['environment']['GROUP_SIZE'] == calls[0][2]['GROUP_SIZE']
    assert record['environment']['CUSTOM_GRADER_MODE'] == 'profile-value'
    assert 'TINKER_API_KEY' not in record['environment']


def test_warmup_error_response_is_fatal_and_restores_accumulator(monkeypatch):
    from skyrl.backends.backward_warmup import run
    from skyrl.tinker.types import ErrorResponse
    original = object()
    slot = SimpleNamespace(mix=None, accum_grads=original, accum_count=3)
    backend = SimpleNamespace(models={'m': slot}, config=SimpleNamespace(train_token_budget=16),
        _row_shard=lambda: 2, _model_pass=lambda *a, **k: {'warmup': ErrorResponse(error='bad kernel', status='failed')})
    monkeypatch.setenv('TUNIX_UNIFORM_SEQ_LEN', '8')
    monkeypatch.setenv('TUNIX_WARMUP_MAX_LENGTH', '8')
    with pytest.raises(RuntimeError, match='bad kernel'):
        run(backend, 'm')
    assert slot.accum_grads is original and slot.accum_count == 3


def test_launch_record_redacts_explicit_tokens_and_equals_urls(tmp_path):
    path = tmp_path / 'launch.json'
    write_launch_contract(path, ['python', '--endpoint=https://user:secret@host/path?auth=secret',
        '--hf-token=secret'], {'HF_TOKEN': 'secret', 'GROUP_SIZE': '32'}, extra_keys=['HF_TOKEN'])
    record = json.loads(path.read_text())
    assert 'secret' not in path.read_text()
    assert record['command'][1] == '--endpoint=https://host/path'
    assert record['environment']['GROUP_SIZE'] == '32'


def test_all_profiles_preserve_legacy_chunked_prefill_emission():
    from tpu.swarm.ray_train.commands import inference_command
    profiles = sorted((ROOT/'tpu/swarm/ray_train/profiles').glob('*.json'))
    assert profiles
    for path in profiles:
        cfg = Config.load(path)
        cmd = inference_command(cfg, Path('/cache'), Path('/source'), Path('/model'), Path('/run'))
        assert '--no-enable-chunked-prefill' not in cmd, path.name
        assert ('--no-enable-prefix-caching' in cmd) == (not cfg.inference.prefix_caching), path.name
        assert ('--enable-chunked-prefill' in cmd) == cfg.inference.chunked_prefill, path.name
