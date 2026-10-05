"""Remote-only trainer: every host trains, every completion comes from leased farms."""
import pytest

from tpu.swarm.ray_train.config import Config

PROFILE = 'tpu/swarm/ray_train/profiles/remote-only-v432-qwen-ac2-pilot-20260922.json'


def remote_only_config(tmp_path, **inference):
    raw = Config.load(PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    raw['inference'].update(inference)
    return Config.from_dict(raw)


def test_remote_only_profile_contract(tmp_path):
    config = remote_only_config(tmp_path)
    assert config.inference.remote_only
    assert config.trainer.hosts == config.hosts == 4
    assert config.trainer.tp * config.trainer.fsdp == 16
    assert config.inference_hosts == 0
    assert config.engine_slots([]) == []
    assert config.borrows_inference
    assert config.inference.external_pool_lease_scope == 'run'
    assert config.inference.external_pool_target_leases == 2
    assert config.inference.external_pool_candidate_limit == 16
    assert config.inference.external_pool_required_compatibility == []
    assert config.client_env['NUM_EPOCHS'] == '1'
    assert not config.bootstrap_layers and not config.bootstrap_max_drafts


def test_candidate_limit_raises_the_two_url_cap(tmp_path):
    urls = [f'http://farm{i}' for i in range(5)]
    config = remote_only_config(tmp_path, external_pool_urls={'Qwen/Qwen3.5-27B': urls})
    assert config.inference.external_pool_urls['Qwen/Qwen3.5-27B'] == urls
    with pytest.raises(ValueError, match='at most 2'):
        remote_only_config(tmp_path, external_pool_candidate_limit=2, external_pool_target_leases=1,
                           external_pool_urls={'Qwen/Qwen3.5-27B': urls[:3]})


@pytest.mark.parametrize('change', [
    dict(external_pool_lease_scope='phase', external_pool_require_initial=False),
    dict(external_pool_updates=False),
    dict(external_pool_scheduler=False),
    dict(external_pool_attestation=False),
    dict(external_pool_target_leases=32),
    dict(external_pool_candidate_limit=0),
    dict(external_pool_required_compatibility=['not-a-digest']),
    dict(external_pool_required_compatibility=['a' * 64, 'a' * 64]),
    dict(request_timeout=7200),
    dict(farm_cancel_grace_seconds=120),
    dict(remote_only='yes'),
])
def test_reject_invalid_remote_only_inference_settings(tmp_path, change):
    with pytest.raises(ValueError):
        remote_only_config(tmp_path, **change)


@pytest.mark.parametrize('change', [
    {'trainer': {'hosts': 2, 'tp': 4, 'fsdp': 2, 'process_bounds': '1,1,2'}},
    {'bootstrap_layers': 1, 'bootstrap_all_hosts': True},
    {'bootstrap_max_drafts': 64},
    {'trainer_env': {'SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC': '3600'}},
    {'client_env': {'TTD_SAMPLING_PROGRESS_TIMEOUT': '900'}},
    {'inference_only': True, 'inference_only_ranks': [0, 1, 2, 3], 'trainer': {'hosts': 0}},
])
def test_reject_invalid_remote_only_topology(tmp_path, change):
    raw = Config.load(PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    for key, value in change.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = dict(raw[key], **value)
        else:
            raw[key] = value
    with pytest.raises(ValueError):
        Config.from_dict(raw)


V5P_PROFILE = 'tpu/swarm/ray_train/profiles/remote-only-v5p32-qwen-ac2-pilot-20260924.json'


def test_v5p32_is_a_valid_remote_only_slice(tmp_path):
    # v5p-32 has the same four-host 2x2x4 layout as v4-32.
    config = Config.load(V5P_PROFILE)
    assert config.accelerator == 'tpu-v5p-32' and config.hosts == 4
    assert config.inference.remote_only and config.trainer.hosts == 4
    assert (config.trainer.tp, config.trainer.fsdp, config.trainer.process_bounds) == (8, 2, '1,1,4')
    assert config.grading_farm_transport


def test_v5p32_pilot_never_reads_another_region():
    from pathlib import Path
    import re
    raw = json.loads(Path(V5P_PROFILE).read_text())
    assert raw['zone'] == 'us-east5-a'
    paths = [raw['bucket'], raw['base_bundle']] + [
        v for v in raw['cache'].values() if isinstance(v, str) and v.startswith('gs://')]
    assert {re.match(r'gs://sk7524-tinker-tpu-(us-[a-z0-9]+)', p).group(1) for p in paths} == {'us-east5'}


@pytest.mark.parametrize('profile', [V5P_PROFILE, 'tpu/swarm/ray_train/profiles/farm-v5p32-qwen-grading-ac2-20260922.json'])
def test_v5p_pilot_profiles_are_locked_to_one_bucket(profile):
    config = Config.load(profile)
    assert config.single_bucket and config.bucket == 'gs://sk7524-tinker-tpu-us-east5'


@pytest.mark.parametrize('field,value,match', [
    ('cache.hf', 'gs://sk7524-tinker-tpu-us-central2/hf-cache-qwen35-v1', 'outside'),
    ('cache.inference_compile_seed', 'gs://sk7524-tinker-tpu-us-east5-other/seed', 'outside'),
    ('base_bundle', 'gs://other-bucket/code.tar.gz', 'outside'),
    ('bucket', 'gs://sk7524-tinker-tpu-us-central2', 'does not match zone'),
    ('bucket', 'gs://my-bucket', 'cannot prove the region'),
])
def test_single_bucket_guard_rejects_any_other_storage(tmp_path, field, value, match):
    raw = Config.load(V5P_PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    if field.startswith('cache.'):
        raw['cache'] = dict(raw['cache'], **{field.split('.', 1)[1]: value})
    else:
        raw[field] = value
    with pytest.raises(ValueError, match=match):
        Config.from_dict(raw)


@pytest.mark.parametrize('accelerator,hosts,zone,ranks', [
    ('tpu-v4-64', 8, 'us-central2-b', 8), ('tpu-v6e-32', 8, 'us-east5-b', 8)])
def test_remote_only_rejects_other_slices(tmp_path, accelerator, hosts, zone, ranks):
    raw = Config.load(PROFILE).to_dict()
    raw.update(root=str(tmp_path), accelerator=accelerator, hosts=hosts, zone=zone)
    raw['trainer'] = dict(raw['trainer'], hosts=ranks)
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_existing_profiles_keep_single_lease_defaults(tmp_path):
    raw = Config.load('tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json').to_dict()
    raw['root'] = str(tmp_path)
    config = Config.from_dict(raw)
    assert not config.inference.remote_only
    assert config.inference.external_pool_target_leases == 1
    assert config.inference.external_pool_candidate_limit == 2
    raw['inference']['external_pool_urls'] = {raw['model']: ['http://a', 'http://b', 'http://c']}
    with pytest.raises(ValueError, match='at most 2'):
        Config.from_dict(raw)


# --- Scheduler: per-farm pools without local engines -----------------------
import asyncio
import json

import httpx

from tpu.swarm.ray_train.hybrid_scheduler import HybridScheduler


def group(n=32):
    return {'choices': [{'token_ids': [1, 2]} for _ in range(n)]}


def test_scheduler_shares_whole_groups_across_two_farms():
    async def run():
        farms = {'a': 2, 'b': 2}
        calls = []
        gate = asyncio.Event()
        router = HybridScheduler(0, 0, lambda: True, farms=lambda: list(farms.items()))

        async def remote(key):
            calls.append(key)
            await gate.wait()
            return group()

        async def local(_):
            raise AssertionError('no local engines')
        payload = {'prompt': [1], 'max_tokens': 10, 'n': 32}
        tasks = [asyncio.create_task(router.run(payload, local, remote)) for _ in range(6)]
        for _ in range(10):
            await asyncio.sleep(0)
        assert sorted(calls) == ['a', 'a', 'b', 'b'] and router.snapshot()['queued'] == 2
        assert router.snapshot()['farms'] == {'a': 2, 'b': 2}
        gate.set()
        results = await asyncio.gather(*tasks)
        assert all(len(r['choices']) == 32 for r in results)
        assert router.snapshot()['remote_active'] == 0 and router.snapshot()['queued'] == 0
        assert router.completed['a'] + router.completed['b'] == 6
        assert router.completed['a'] >= 2 and router.completed['b'] >= 2
    asyncio.run(run())


def test_scheduler_requeues_failed_remote_group_on_another_farm_without_local():
    async def run():
        farms = {'a': 1, 'b': 1}
        failed = []
        calls = []
        router = HybridScheduler(0, 0, lambda: True, farms=lambda: list(farms.items()),
                                 on_remote_failure=failed.append)

        async def remote(key):
            calls.append(key)
            if key == 'a':
                farms.pop('a')  # The farm is ineligible until it re-attests.
                return None
            return group(4)
        router.rates.update(a=100, b=1)  # Prefer 'a' first.
        result = await router.run({'prompt': [1], 'max_tokens': 1, 'n': 4},
                                  lambda _: None, remote)
        assert calls == ['a', 'b'] and failed == ['a'] and router.retries == 1
        assert len(result['choices']) == 4
        assert 'a' not in router.pools and not any(router.pools['b'])
    asyncio.run(run())


def test_scheduler_waits_indefinitely_and_alerts_while_no_farm_is_eligible():
    async def run():
        farms = {}
        events = []
        router = HybridScheduler(0, 0, lambda: True, lambda event, **f: events.append((event, f)),
                                 farms=lambda: list(farms.items()), alert_seconds=0)

        async def remote(key):
            return group(1)
        task = asyncio.create_task(router.run({'prompt': [1], 'n': 1}, lambda _: None, remote))
        await asyncio.sleep(.01)
        assert not task.done() and router.snapshot()['queued'] == 1
        assert any(e == 'remote_queue_waiting' and f['farms'] == [] for e, f in events)
        farms['late'] = 1
        async with router.condition:
            router.condition.notify_all()
        result = await asyncio.wait_for(task, 5)
        assert len(result['choices']) == 1 and router.completed['late'] == 1
    asyncio.run(run())


def test_scheduler_releases_slot_by_identity_after_pool_compaction():
    async def run():
        farms = {'a': 2}
        gates = {0: asyncio.Event(), 1: asyncio.Event()}
        calls = []
        router = HybridScheduler(0, 0, lambda: True, farms=lambda: list(farms.items()))

        async def remote(key):
            index = len(calls)
            calls.append(key)
            await gates[index].wait()
            return group(1)
        payload = {'prompt': [1], 'n': 1}
        first = asyncio.create_task(router.run(payload, lambda _: None, remote))
        second = asyncio.create_task(router.run(payload, lambda _: None, remote))
        for _ in range(10):
            await asyncio.sleep(0)
        gates[0].set()
        await first
        farms['a'] = 1  # Compaction moves the second job to slot 0.
        router._remote_pools()
        assert router.pools['a'][0] is not None and len(router.pools['a']) == 1
        gates[1].set()
        await second
        assert router.pools['a'] == [None]
    asyncio.run(run())


# --- Borrower request-failure policy ---------------------------------------

from tpu.swarm.ray_train.borrowing import Borrower, RemoteRequestRejected
from tests.tpu_swarm.test_inference_borrowing import Farm, config as borrowing_config


class StatusFarm(Farm):
    """Farm fake whose /v1/completions status is programmable per test."""

    def __init__(self):
        super().__init__()
        self.generate_status = None
        self.capabilities = None
        self.abort_choices = False

    async def __call__(self, request):
        path = request.url.path
        if path == '/v1/completions' and self.generate_status is not None:
            self.requests.append((request.url.host, path))
            return httpx.Response(self.generate_status, json={'detail': 'rejected'})
        if path == '/v1/completions' and self.abort_choices:
            self.requests.append((request.url.host, path))
            n = json.loads(request.content).get('n', 1)
            return httpx.Response(200, json={'choices': [
                {'index': i, 'token_ids': [3], 'logprobs': {'token_logprobs': [-.1]},
                 'finish_reason': 'abort' if i == 0 else 'stop'} for i in range(n)]})
        if path == '/status' and 'X-Lease-ID' not in request.headers and self.capabilities:
            self.requests.append((request.url.host, path))
            return httpx.Response(200, json={'instance': self.incarnation,
                                             'capabilities': {'compatibility_sha256': self.capabilities}})
        return await super().__call__(request)


class ProbeBorrower(Borrower):
    request_failure_policy = 'probe'


def probe_case(tmp_path, operation, **inference):
    async def run():
        raw = borrowing_config(tmp_path).to_dict()
        raw['inference'].update(inference)
        raw['inference']['external_pool_heartbeat_seconds'] = 1
        raw['inference']['external_pool_rpc_timeout'] = 1
        raw['inference']['external_pool_lease_seconds'] = 30
        cfg = Config.from_dict(raw)
        archive = tmp_path / 'adapter.tar'
        archive.write_bytes(b'adapter bytes')
        farm = StatusFarm()
        events = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(farm)) as client:
            borrower = ProbeBorrower(cfg, client, lambda event, **fields: events.append((event, fields)))
            try:
                await operation(borrower, farm, archive, events)
            finally:
                await borrower.end()
    asyncio.run(run())


PAYLOAD = {'model': 'Qwen/Qwen3.5-27B', 'prompt': [1, 2], 'n': 2, 'max_tokens': 4}


def test_probe_policy_5xx_pauses_then_renewal_recovers_without_losing_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        lease = borrower.lease
        assert lease and borrower._eligible(lease)
        farm.generate_status = 503
        farm.heartbeat_gate = asyncio.Event()  # Hold the probe renewal to observe the pause.
        farm.heartbeat_seen.clear()
        assert await borrower.generate(PAYLOAD, 0, 0, prefer_remote=True) is None
        assert not lease.lost.is_set() and not lease.ready and not borrower._eligible(lease)
        assert [e for e, _ in events if e == 'borrow_health_paused']
        await asyncio.wait_for(farm.heartbeat_seen.wait(), 5)  # Woken by the probe, not the interval.
        farm.generate_status = None
        farm.heartbeat_gate.set()
        for _ in range(50):
            if lease.ready:
                break
            await asyncio.sleep(.05)
        assert lease.ready and borrower._eligible(lease)
        assert [e for e, _ in events if e == 'borrow_health_recovered']
        assert borrower.lease is lease
        result = await borrower.generate(PAYLOAD, 0, 0, prefer_remote=True)
        assert result and len(result['choices']) == 2
    probe_case(tmp_path, operation)


def test_probe_policy_409_loses_only_this_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        lease = borrower.lease
        farm.generate_status = 409
        assert await borrower.generate(PAYLOAD, 0, 0, prefer_remote=True) is None
        assert lease.lost.is_set()
        assert any(f.get('reason') == 'lease rejected' for e, f in events if e == 'borrow_heartbeat_failed')
    probe_case(tmp_path, operation)


def test_probe_policy_aborted_completion_is_retried_not_returned(tmp_path):
    # A preempted farm's vLLM answered 200 with finish_reason 'abort'; the
    # trainer API rejected those sequences and the client dropped the groups.
    async def operation(borrower, farm, archive, events):
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        lease = borrower.lease
        farm.abort_choices = True
        farm.heartbeat_gate = asyncio.Event()  # Hold the probe renewal to observe the pause.
        assert await borrower.generate(PAYLOAD, 0, 0, prefer_remote=True) is None
        assert not lease.lost.is_set() and not borrower._eligible(lease)
        assert [e for e, _ in events if e == 'borrow_health_paused']
        assert not [e for e, _ in events if e == 'borrow_generated']
        farm.heartbeat_gate.set()
    probe_case(tmp_path, operation)


def test_probe_policy_4xx_rejection_propagates_without_lease_change(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        lease = borrower.lease
        farm.generate_status = 422
        with pytest.raises(RemoteRequestRejected) as info:
            await borrower.generate(PAYLOAD, 0, 0, prefer_remote=True)
        assert info.value.status_code == 422
        assert not lease.lost.is_set() and lease.ready and lease.active == 0
    probe_case(tmp_path, operation)


def test_contract_hooks_accept_a_set_and_claim_the_farm_hash(tmp_path):
    class SetBorrower(ProbeBorrower):
        accepted = {'b' * 64, 'c' * 64}

        def _contract_accepted(self, sha):
            return sha in self.accepted

        def _contract_claim(self, sha):
            return sha

    async def run():
        raw = borrowing_config(tmp_path).to_dict()
        raw['inference']['external_pool_attestation'] = True
        cfg = Config.from_dict(raw)
        farm = StatusFarm()
        farm.capabilities = 'c' * 64
        claims = []

        async def spy(request):
            if request.url.path == '/acquire_lease':
                claims.append(json.loads(request.content).get('compatibility_sha256'))
            return await farm(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(spy)) as client:
            borrower = SetBorrower(cfg, client)
            await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', None)
            assert borrower.lease is not None
            assert claims[0] == 'c' * 64
            await borrower.end()
            farm.capabilities = 'd' * 64
            events = []
            borrower = SetBorrower(cfg, client, lambda event, **fields: events.append(event))
            await borrower.begin('phase-2', 'Qwen/Qwen3.5-27B', None)
            assert borrower.lease is None and 'borrow_incompatible_runtime' in events
            await borrower.end()
    asyncio.run(run())


# --- MultiRunBorrower --------------------------------------------------------
from tpu.swarm.ray_train.multi_borrowing import MultiRunBorrower


class MultiFarm(Farm):
    """Per-host programmable capabilities and completion status."""

    def __init__(self):
        super().__init__()
        self.capabilities = {}
        self.generate_status = {}
        self.acquires = {}
        self.engines = {}  # host -> engine count (default 4)
        self.current_host = None

    def status(self, lease):
        result = super().status(lease)
        engines = self.engines.get(self.current_host, 4)
        result.update(expected_engines=engines, ready_engines=engines if result['adapter_sha256'] else 0)
        return result

    async def __call__(self, request):
        host, path = request.url.host, request.url.path
        self.current_host = host
        if path == '/acquire_lease' and host not in self.down and not json.loads(request.content).get('lease_id'):
            self.acquires[host] = self.acquires.get(host, 0) + 1
        if path.endswith('/upload_lora_adapter') and host in self.engines and host not in self.down:
            response = await super().__call__(request)
            body = response.json()
            body['loaded'] = [str(i) for i in range(self.engines[host])]
            return httpx.Response(200, json=body)
        if path == '/v1/completions' and host in self.generate_status and host not in self.down:
            self.requests.append((host, path))
            return httpx.Response(self.generate_status[host], json={'detail': 'rejected'})
        if path == '/status' and 'X-Lease-ID' not in request.headers and host in self.capabilities:
            self.requests.append((host, path))
            return httpx.Response(200, json={'instance': self.incarnation,
                                             'capabilities': {'compatibility_sha256': self.capabilities[host]}})
        return await super().__call__(request)


async def until(predicate, timeout=8):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError('condition not reached in time')
        await asyncio.sleep(.02)


def multi_case(tmp_path, operation, urls=('http://farm1', 'http://farm2'), **inference):
    async def run():
        raw = borrowing_config(tmp_path).to_dict()
        raw['inference'].update(
            external_pool_urls={raw['model']: list(urls)}, external_pool_updates=True,
            external_pool_lease_scope='run', external_pool_require_initial=True,
            external_pool_target_leases=2, external_pool_candidate_limit=8,
            external_pool_heartbeat_seconds=1, external_pool_rpc_timeout=1,
            external_pool_health_grace_seconds=1, external_pool_lease_seconds=30,
            external_pool_prepare_timeout=20)
        raw['inference'].update(inference)
        cfg = Config.from_dict(raw)
        archive = tmp_path / 'adapter.tar'
        archive.write_bytes(b'adapter bytes v1')
        farm = MultiFarm()
        events = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(farm)) as client:
            borrower = MultiRunBorrower(cfg, client, lambda event, **fields: events.append((event, fields)))
            try:
                await operation(borrower, farm, archive, events)
            finally:
                await borrower.close()
            assert 'secret-' not in json.dumps(events)
            assert not any('token' in json.dumps(f) for e, f in events)
    asyncio.run(run())


def held(borrower):
    return sorted(m.url for m in borrower.held())


def eligible(borrower):
    return sorted(m.url for m in borrower.eligible_members())


def test_multi_borrower_holds_two_leases_and_shares_generation(tmp_path):
    async def operation(borrower, farm, archive, events):
        snapshot = await borrower.reserve()
        assert snapshot['reserved'] and held(borrower) == ['http://farm1', 'http://farm2']
        assert snapshot['held'] == ['http://farm1', 'http://farm2'] and snapshot['target'] == 2
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        assert borrower.eligible_pools() == [('http://farm1', 4), ('http://farm2', 4)]
        for farm_key in ('http://farm1', 'http://farm2', None, None):
            result = await borrower.generate(PAYLOAD, farm=farm_key)
            assert result and len(result['choices']) == 2
        hosts = {h for h, p in farm.requests if p == '/v1/completions'}
        assert hosts == {'farm1', 'farm2'}
        assert borrower.snapshot()['ready'] and borrower.lease is not None
        assert borrower._eligible(borrower.lease)
        await borrower.end('phase-1')
        assert borrower.phase is None and held(borrower) == ['http://farm1', 'http://farm2']
    multi_case(tmp_path, operation)


def test_multi_borrower_farm_loss_keeps_the_other_farm_serving(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.down.add('farm1')
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None
        await until(lambda: held(borrower) == ['http://farm2'])
        assert eligible(borrower) == ['http://farm2']
        result = await borrower.generate(PAYLOAD)
        assert result and farm.requests[-1] == ('farm2', '/v1/completions')
        assert any(e == 'borrow_heartbeat_failed' and f['service'] == 'http://farm1' for e, f in events)
        # A recovered farm must republish the phase adapter before it serves again.
        farm.owners.pop('farm1', None)
        borrower.members['http://farm1'].uncertain_until = 0
        farm.down.discard('farm1')
        await until(lambda: eligible(borrower) == ['http://farm1', 'http://farm2'])
        lease = borrower.members['http://farm1'].lease
        assert lease.digest == farm.owners['farm1']['adapter_sha256'] and lease.adapter == farm.owners['farm1']['adapter_name']
    multi_case(tmp_path, operation)


def test_multi_borrower_all_farms_lost_then_replacement_serves_after_attestation(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.down.update({'farm1', 'farm2'})
        await until(lambda: held(borrower) == [])
        assert borrower.eligible_pools() == [] and await borrower.generate(PAYLOAD) is None
        assert not borrower.snapshot()['ready'] and borrower.snapshot()['reserved'] is False
        await until(lambda: all(m.uncertain_until > 0 for m in borrower.members.values()))
        # Unacknowledged releases fence each farm for one lease TTL; simulate
        # the farm-side expiry (owners cleared) and the TTL elapsing.
        farm.owners.clear()
        for member in borrower.members.values():
            member.uncertain_until = 0
        farm.down.discard('farm2')
        await until(lambda: held(borrower) == ['http://farm2'])
        await until(lambda: eligible(borrower) == ['http://farm2'])
        ready = [f for e, f in events if e == 'borrow_ready' and f['service'] == 'http://farm2'
                 and f['phase'] == 'phase-1']
        assert len(ready) == 2 and farm.owners['farm2']['adapter_sha256'] == ready[-1]['sha256']
        member = borrower.members['http://farm2']
        assert member.lease.digest == ready[-1]['sha256'] and member.lease.adapter.startswith('borrow-')
        result = await borrower.generate(PAYLOAD)
        assert result and len(result['choices']) == 2
        assert farm.requests[-1] == ('farm2', '/v1/completions')
    multi_case(tmp_path, operation)


def test_multi_borrower_never_exceeds_target_and_never_double_acquires(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        assert held(borrower) == ['http://farm1', 'http://farm2']
        assert farm.acquires == {'farm1': 1, 'farm2': 1}
        await borrower.reserve()
        await asyncio.sleep(1.5)  # A few reconcile ticks.
        assert farm.acquires == {'farm1': 1, 'farm2': 1}
        farm.down.add('farm1')
        await until(lambda: held(borrower) == ['http://farm2', 'http://farm3'])
        assert farm.acquires['farm3'] == 1 and farm.acquires['farm2'] == 1
    multi_case(tmp_path, operation, urls=('http://farm1', 'http://farm2', 'http://farm3'))


def test_multi_borrower_update_urls_keeps_live_leases_and_adds_candidates(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        assert held(borrower) == ['http://farm1']
        borrower.update_urls('Qwen/Qwen3.5-27B', ['http://farm2', 'http://farm3'])
        assert borrower.urls == ['http://farm2', 'http://farm3']
        await until(lambda: held(borrower) == ['http://farm1', 'http://farm2'])
        assert 'http://farm1' in borrower.members  # Live lease survives its removal from the list.
        with pytest.raises(Exception):
            borrower.update_urls('Qwen/Qwen3.5-27B', ['http://farm%d' % i for i in range(9)])
    multi_case(tmp_path, operation, urls=('http://farm1',))


def test_multi_borrower_learns_the_first_contract_and_rejects_others(tmp_path):
    async def operation(borrower, farm, archive, events):
        farm.capabilities.update(farm1='c' * 64, farm2='d' * 64)
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: borrower.accepted == {'c' * 64})
        await until(lambda: held(borrower) == ['http://farm1'])
        await asyncio.sleep(1.5)
        assert held(borrower) == ['http://farm1'] and borrower.required_contract == 'c' * 64
        assert any(e == 'borrow_incompatible_runtime' and f['service'] == 'http://farm2' for e, f in events)
        assert borrower.snapshot()['accepted_compatibility'] == ['c' * 64]
    multi_case(tmp_path, operation, external_pool_attestation=True)


def test_multi_borrower_configured_contract_list_accepts_both_hashes(tmp_path):
    async def operation(borrower, farm, archive, events):
        farm.capabilities.update(farm1='c' * 64, farm2='d' * 64)
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        assert borrower.required_contract is None and borrower.accepts('c' * 64) and not borrower.accepts('e' * 64)
    multi_case(tmp_path, operation, external_pool_attestation=True,
               external_pool_required_compatibility=['c' * 64, 'd' * 64])


def test_multi_borrower_request_failure_does_not_drop_a_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.generate_status['farm1'] = 503
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None
        assert held(borrower) == ['http://farm1', 'http://farm2']
        del farm.generate_status['farm1']
        await until(lambda: len(eligible(borrower)) == 2)
        farm.generate_status['farm1'] = 422
        with pytest.raises(RemoteRequestRejected):
            await borrower.generate(PAYLOAD, farm='http://farm1')
        assert eligible(borrower) == ['http://farm1', 'http://farm2']
    multi_case(tmp_path, operation)


def test_multi_borrower_mixes_farm_sizes_learned_at_acquire(tmp_path):
    """A v6e-32 farm (8 engines) and a v4-32 farm (4) serve one trainer."""
    async def operation(borrower, farm, archive, events):
        farm.engines['farm1'] = 8
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        assert borrower.members['http://farm1'].lease.engines == 8
        assert borrower.members['http://farm2'].lease.engines == 4
        assert borrower.snapshot()['engines'] == 12
        ready = {f['service']: f['engines'] for e, f in events if e == 'borrow_ready' and f['phase'] == 'phase-1'}
        assert ready == {'http://farm1': 8, 'http://farm2': 4}
        # A farm that changes size under a live lease loses attestation; the
        # reconcile loop then re-acquires it at the new size and republishes.
        farm.engines['farm2'] = 6
        await until(lambda: borrower.members['http://farm2'].lease.engines == 6
                    and 'http://farm2' in eligible(borrower))
        assert any(e == 'borrow_heartbeat_failed' and f['service'] == 'http://farm2'
                   and f['reason'] == 'unexpected farm size' for e, f in events)
        republished = [f for e, f in events if e == 'borrow_ready' and f['service'] == 'http://farm2'
                       and f['phase'] == 'phase-1']
        assert [f['engines'] for f in republished] == [4, 6]
    multi_case(tmp_path, operation)


def test_legacy_borrower_still_requires_the_configured_farm_size(tmp_path):
    async def run():
        raw = borrowing_config(tmp_path).to_dict()
        cfg = Config.from_dict(raw)
        farm = MultiFarm()
        farm.engines['farm1'] = 8
        farm.engines['farm2'] = 8
        events = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(farm)) as client:
            borrower = Borrower(cfg, client, lambda event, **fields: events.append(event))
            await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', None)
            assert borrower.lease is None and 'borrow_unavailable' in events
            await borrower.end()
    asyncio.run(run())


def test_multi_borrower_close_releases_every_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        assert set(farm.owners) == {'farm1', 'farm2'}
        await borrower.close()
        assert farm.owners == {} and borrower.closed
        assert any(e == 'borrow_closed' for e, _ in events)
        assert borrower.snapshot()['reserved'] is False
    multi_case(tmp_path, operation)


def test_discovery_group_hides_trainer_from_other_supervisors(tmp_path, monkeypatch):
    """A second supervisor watching the pool must not be able to feed the trainer farms.

    2026-09-24: another run's supervisor pushed its farms (one in us-central2)
    to the v5p pilot trainer because the trainer answered every supervisor.
    """
    import inspect
    from types import SimpleNamespace
    from tpu.swarm.ray_train import serving
    from tests.tpu_swarm.test_farm_leases import Remote

    async def run():
        raw = Config.load(PROFILE).to_dict()
        raw['root'] = str(tmp_path)
        raw['inference'].update(external_pool_discovery_group='remote-only-pilot')
        cfg = Config.from_dict(raw)
        snapshot = dict(fatal_error=None, version=None, versions=[], expected=[], replicas=[], starts={}, exhausted=[])
        catalog = SimpleNamespace(snapshot=Remote(lambda: snapshot), commit=Remote(lambda *a: None))
        gateway = serving.Ingress.func_or_class.__mro__[1](cfg.to_dict(), [], catalog, [])
        client = httpx.AsyncClient(transport=httpx.MockTransport(MultiFarm()))
        await gateway.http.aclose()
        gateway.http = client
        gateway.borrower = MultiRunBorrower(cfg, client, lambda event, **fields: None)
        monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
        push = dict(model=cfg.model, run_id=cfg.run_id, instance=gateway.borrowing_instance,
                    urls=['http://other-run-farm'])
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as http:
                path = '/skyrl/v1/borrowing/services'
                anonymous = (await http.get(path)).json()
                assert anonymous['enabled'] is False and anonymous['discovery_group_required'] is True
                assert (await http.get(path, params={'group': 'hybrid'})).json()['enabled'] is False
                assert (await http.get(path, params={'group': 'remote-only-pilot'})).json()['enabled'] is True
                # The instance is still readable: the controller's admission loop needs it.
                assert anonymous['instance'] == gateway.borrowing_instance
                assert (await http.post(path, json=push)).status_code == 403
                assert (await http.post(path, json=dict(push, group='hybrid'))).status_code == 403
                assert gateway.borrower.urls == []
                own = await http.post(path, json=dict(push, group='remote-only-pilot', urls=['http://own-farm']))
                assert own.status_code == 200 and own.json()['enabled'] is True
                assert list(gateway.borrower.urls) == ['http://own-farm']
        finally:
            await gateway.borrower.close()
            await client.aclose()
    asyncio.run(run())


def test_supervisor_presents_its_discovery_group(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import borrowing_supervisor as supervisor
    (tmp_path / 'trainer-1').write_text('Host trainer-1\n')
    sent = []

    def fake_run(argv, input, **kwargs):
        sent.append(input.split('\n', 1)[0])
        return SimpleNamespace(returncode=0, stdout='{"ok": true, "result": {}}\n', stderr='')
    from types import SimpleNamespace
    monkeypatch.setattr(supervisor.subprocess, 'run', fake_run)
    supervisor.rpc(tmp_path, 'trainer-1', 'target', 24800, group='remote-only-pilot')
    supervisor.rpc(tmp_path, 'trainer-1', 'target', 24800, {'urls': []}, group='remote-only-pilot')
    supervisor.rpc(tmp_path, 'trainer-1', 'farm', 24800, group='remote-only-pilot')
    supervisor.rpc(tmp_path, 'trainer-1', 'target', 24800)
    read, push, farm, legacy = [eval(line.removeprefix('PARAMS = ')) for line in sent]
    assert read['query'] == '?group=remote-only-pilot' and 'body' not in read
    assert push['body'] == {'urls': [], 'group': 'remote-only-pilot'}
    assert 'query' not in farm
    assert 'query' not in legacy and 'body' not in legacy


def test_zero_engine_ingress_serves_control_plane_and_routes_groups_to_farms(tmp_path, monkeypatch):
    import inspect
    from types import SimpleNamespace
    from tpu.swarm.ray_train import serving
    from tests.tpu_swarm.test_farm_leases import Remote

    async def run():
        raw = Config.load(PROFILE).to_dict()
        raw['root'] = str(tmp_path)
        raw['inference'].update(external_pool_urls={raw['model']: ['http://farm1', 'http://farm2']},
                                external_pool_heartbeat_seconds=1, external_pool_rpc_timeout=1,
                                external_pool_health_grace_seconds=1, external_pool_lease_seconds=30,
                                external_pool_prepare_timeout=20, external_pool_queue_alert_seconds=1)
        cfg = Config.from_dict(raw)
        snapshot = dict(fatal_error=None, version=None, versions=[], expected=[], replicas=[], starts={}, exhausted=[])
        catalog = SimpleNamespace(snapshot=Remote(lambda: snapshot), commit=Remote(lambda *a: None))
        gateway = serving.Ingress.func_or_class.__mro__[1](cfg.to_dict(), [], catalog, [])
        assert gateway.remote_only and gateway.engines == [] and gateway.scheduler.local == []
        farm = MultiFarm()
        farm.capabilities.update(farm1='c' * 64, farm2='c' * 64)
        events = []
        client = httpx.AsyncClient(transport=httpx.MockTransport(farm))
        await gateway.http.aclose()
        gateway.http = client
        gateway.borrower = MultiRunBorrower(cfg, client, lambda event, **fields: events.append((event, fields)))
        gateway.scheduler.farms = gateway.borrower.eligible_pools
        gateway.scheduler.report = lambda event, **fields: events.append((event, fields))
        monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as http:
                health = await http.get('/health')
                assert health.status_code == 200 and health.json()['remote_only']
                status = (await http.get('/status')).json()
                assert status['remote_only'] and status['leases'] == [] and status['capabilities']['engines'] == 0
                assert status['capabilities']['accepted_compatibility'] == []
                services = (await http.get('/skyrl/v1/borrowing/services')).json()
                assert services['remote_only'] and services['target_leases'] == 2 and services['candidate_limit'] == 16
                assert (await http.post('/tokenize', json={'model': cfg.model, 'prompt': 'x'})).status_code == 503
                farms = await http.get('/skyrl/v1/grading/farms')
                assert farms.status_code == 200 and farms.json()['farms'] == []
                # Groups queue while no farm is held; the alert fires; nothing fails.
                payload = dict(model=cfg.model, prompt=[1, 2], n=2, max_tokens=4)
                pending = asyncio.create_task(http.post('/v1/completions', json=payload))
                await asyncio.sleep(.05)
                assert not pending.done() and gateway.scheduler.snapshot()['queued'] == 1
                await until(lambda: any(e == 'remote_queue_waiting' for e, _ in events), 5)
                reservation = await http.post('/skyrl/v1/borrowing/reservation',
                    json=dict(run_id=cfg.run_id, instance=gateway.borrowing_instance, action='acquire'))
                assert reservation.status_code == 200 and sorted(reservation.json()['held']) == ['http://farm1', 'http://farm2']
                assert reservation.json()['accepted_compatibility'] == ['c' * 64]
                assert (await http.get('/status')).json()['capabilities']['compatibility_sha256'] == 'c' * 64
                farms = (await http.get('/skyrl/v1/grading/farms')).json()['farms']
                assert sorted(f['url'] for f in farms) == ['http://farm1', 'http://farm2']
                assert all(f['token'].startswith('lease-') and f['capacity'] == 4 for f in farms)
                begin = await http.post('/skyrl/v1/borrowing/begin',
                    json={'phase_id': 'phase1', 'bootstrap': True, 'expected_n': 2})
                assert begin.status_code == 200 and begin.json()['remote_only']
                response = await asyncio.wait_for(pending, 10)
                assert response.status_code == 200 and len(response.json()['choices']) == 2
                responses = await asyncio.gather(*(http.post('/v1/completions', json=payload) for _ in range(6)))
                assert all(r.status_code == 200 for r in responses)
                hosts = {h for h, p in farm.requests if p == '/v1/completions'}
                assert hosts == {'farm1', 'farm2'} and gateway.active == 0
                status = (await http.get('/status')).json()
                assert sorted(l['url'] for l in status['leases']) == ['http://farm1', 'http://farm2']
                assert status['scheduling']['farms'] == {'http://farm1': 0, 'http://farm2': 0}
                # A farm-side 4xx surfaces as the same status to the API server.
                farm.generate_status['farm1'] = 422
                farm.generate_status['farm2'] = 422
                rejected = await http.post('/v1/completions', json=payload)
                assert rejected.status_code == 422
                farm.generate_status.clear()
                ended = await http.post('/skyrl/v1/borrowing/end', json={'phase_id': 'phase1'})
                assert ended.status_code == 200
                closed = await http.post('/skyrl/v1/borrowing/reservation',
                    json=dict(run_id=cfg.run_id, instance=gateway.borrowing_instance, action='close'))
                assert closed.status_code == 200 and farm.owners == {}
        finally:
            await gateway.borrower.close()
            await client.aclose()
    asyncio.run(run())


def test_expired_lease_cancels_inflight_work_and_frees_the_farm(tmp_path, monkeypatch):
    """Farm side: an abruptly lost trainer must not quarantine the farm."""
    from dataclasses import replace
    from tests.tpu_swarm.test_farm_leases import farm as deployed_farm, Remote

    async def run():
        async with deployed_farm(tmp_path, monkeypatch) as (client, gateway, _, __, entered, finish):
            gateway.config = replace(gateway.config, inference=replace(
                gateway.config.inference, farm_cancel_grace_seconds=0, farm_drain_timeout=5))
            quarantined = asyncio.Event()
            gateway.catalog.quarantine = Remote(lambda reason: quarantined.set())
            engine_cancels = []
            gateway.engines[0].cancel_inflight = Remote(lambda: engine_cancels.append(1) or {'cancelled': 1})
            lease = (await client.post('/acquire_lease', json={'owner_run': 'lost-trainer'})).json()
            request = asyncio.create_task(client.post('/v1/completions',
                json={'model': gateway.config.model, 'block': True}, headers={'X-Lease-ID': lease['lease_id']}))
            await entered.wait()
            assert gateway.active == 1 and len(gateway.inflight_leases) == 1
            gateway.lease['deadline'] = 0
            response = await asyncio.wait_for(request, 5)
            assert response.status_code == 409 and 'cancelled' in response.text
            await until(lambda: gateway.active == 0 and not gateway.inflight, 3)
            assert engine_cancels == [1] and not quarantined.is_set()
            events = [json.loads(line) for line in (gateway.run / 'inference-events.jsonl').read_text().splitlines()]
            cancelled = [e for e in events if e['event'] == 'lease_inflight_cancelled']
            assert cancelled and cancelled[0]['requests'] == 1 and cancelled[0]['owner_run'] == 'lost-trainer'
            # The farm is immediately leasable by a new owner; no relaunch needed.
            granted = await client.post('/acquire_lease', json={'owner_run': 'next-trainer'})
            assert granted.status_code == 200 and (await client.get('/health')).status_code == 200
            assert gateway.lease['owner_run'] == 'next-trainer'
            finish.set()
    asyncio.run(run())


# --- Controller ------------------------------------------------------------
from tpu.swarm.select_v4_32_topology import full_slice_rank_order, slice_rows, verify_full_slice_order


def slice_records(rank_to_z):
    return [dict(process_id=rank, coords=[[x, y, z] for x in range(2) for y in range(2)])
            for rank, z in enumerate(rank_to_z)]


def test_v4_32_full_slice_order_guard():
    assert verify_full_slice_order(slice_records([0, 1, 2, 3])) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='not in physical z order'):
        verify_full_slice_order(slice_records([0, 2, 1, 3]))
    with pytest.raises(ValueError, match='complete v4-32'):
        verify_full_slice_order(slice_records([0, 1, 2, 2]))
    with pytest.raises(ValueError, match='expected Sky ranks'):
        verify_full_slice_order(slice_records([0, 1, 2]))


@pytest.mark.parametrize('rank_to_z', [[0, 1, 2, 3], [1, 2, 3, 0], [3, 2, 1, 0], [2, 0, 3, 1]])
def test_v4_32_trainer_ranks_follow_physical_rows(rank_to_z):
    # [1, 2, 3, 0] is what a pool worker actually reported on 2026-09-24.
    records = slice_records(rank_to_z)
    assert slice_rows(records) == rank_to_z
    train_ranks = full_slice_rank_order(records)
    # Process i (position in train_ranks) is the host on physical row i.
    assert [rank_to_z[rank] for rank in train_ranks] == [0, 1, 2, 3]
    # The subset re-probe reports process ids in train_ranks order; once
    # reordered, process i owns row i and the verification passes.
    subset = [dict(process_id=process, coords=records[rank]['coords']) for process, rank in enumerate(train_ranks)]
    assert verify_full_slice_order(subset) == [0, 1, 2, 3]


def test_v4_32_reorder_still_rejects_broken_slices():
    with pytest.raises(ValueError, match='complete v4-32'):
        full_slice_rank_order(slice_records([0, 1, 1, 3]))
    straddle = slice_records([0, 1, 2, 3])
    straddle[0]['coords'][0] = [0, 0, 1]
    straddle[1]['coords'][0] = [0, 0, 0]
    with pytest.raises(ValueError, match='spans several z rows'):
        full_slice_rank_order(straddle)


def controller_for(tmp_path):
    from tpu.swarm.ray_train import controller
    raw = Config.load(PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    cfg = Config.from_dict(raw)
    owner = controller.Controller(cfg, ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4'])
    events = []
    owner.report = lambda event, **kw: events.append((event, kw))
    return controller, owner, events


def test_controller_waits_indefinitely_for_a_farm_in_remote_only(tmp_path, monkeypatch):
    controller, owner, events = controller_for(tmp_path)
    now, acquires = [1000.0], []

    def transport(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'instance': 'control-plane'})
        body = json.loads(request.content)
        assert body['action'] == 'acquire' and body['instance'] == 'control-plane'
        acquires.append(now[0])
        # Roughly 25 minutes without any farm, then one appears.
        return httpx.Response(200, json={'reserved': now[0] - 1000 > 1500, 'held': ['http://farm-a'],
                                         'candidates': ['http://farm-a'], 'target': 2})
    original_client = httpx.Client
    monkeypatch.setattr(controller.httpx, 'Client', lambda **kw: original_client(
        transport=httpx.MockTransport(transport), **kw))
    monkeypatch.setattr(controller.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(owner.stopping, 'wait', lambda seconds: now.__setitem__(0, now[0] + seconds))
    owner.wait_farm_admission()
    names = [e for e, _ in events]
    assert names[0] == 'waiting_for_farm' and events[0][1]['unbounded']
    assert 'farm_admission_local_fallback' not in names
    assert names[-1] == 'farm_admitted' and events[-1][1]['held'] == ['http://farm-a']
    waits = [kw for e, kw in events if e == 'farm_admission_waiting']
    assert len(waits) >= 20 and all(kw['candidates'] == ['http://farm-a'] for kw in waits)
    assert len(acquires) > 300 and now[0] - 1000 > 1500  # Far beyond the legacy 300 s deadline.


def test_controller_reports_remote_lease_changes_without_failing(tmp_path, monkeypatch):
    controller, owner, events = controller_for(tmp_path)
    owner.local_inference_instance = 'control-plane'
    held = [[]]
    now = [5000.0]

    def transport(request):
        assert request.url.path == '/status'
        return httpx.Response(200, json={'instance': 'control-plane', 'fatal_error': None, 'exhausted': [],
                                         'borrowing': {'held': held[0], 'target': 2, 'candidates': ['u']},
                                         'scheduling': {'queued': 3}})
    original_client = httpx.Client
    monkeypatch.setattr(controller.httpx, 'Client', lambda **kw: original_client(
        transport=httpx.MockTransport(transport), **kw))
    monkeypatch.setattr(controller.time, 'monotonic', lambda: now[0])
    owner.check_local_inference()
    assert owner.failure is None
    assert [e for e, _ in events] == ['remote_leases', 'remote_leases_zero']
    assert events[0][1] == dict(held=[], target=2, candidates=['u'], queued=3)
    now[0] += 10
    owner.check_local_inference()
    assert [e for e, _ in events] == ['remote_leases', 'remote_leases_zero']  # Alert cadence respected.
    now[0] += 60
    held[0] = ['http://farm-b', 'http://farm-a']
    owner.check_local_inference()
    assert events[-1] == ('remote_leases', dict(held=['http://farm-a', 'http://farm-b'], target=2,
                                                candidates=['u'], queued=3))
    assert owner.failure is None


# --- Supervisor admission ---------------------------------------------------
from tpu.swarm.ray_train.farm_admission import assignments, candidate_limit


def farm_row(job_id, url, owner=None, state='unleased', sha='c' * 64, name='inference-farm-v4-32'):
    return dict(job_id=job_id, url=url, models=['qwen'], owner_run=owner, state=state,
                active=0, source_name=name, capabilities={'compatibility_sha256': sha})


def test_multi_lease_target_gets_owned_plus_free_candidates_up_to_limit():
    farms = [farm_row(1, 'http://a', owner='pilot:x', state='ready'),
             farm_row(2, 'http://b', owner='pilot:x', state='ready'),
             farm_row(3, 'http://c'), farm_row(4, 'http://d'), farm_row(5, 'http://e', sha='d' * 64),
             farm_row(6, 'http://other-owner', owner='someone:y', state='ready')]
    target = dict(model='qwen', run_id='pilot', lease_scope='run', target_leases=2, candidate_limit=4,
                  accepted_compatibility=['c' * 64])
    legacy = dict(model='qwen', run_id='legacy', lease_scope='run')
    result = assignments([], farms, {10: target, 11: legacy})
    assert candidate_limit(target) == 4 and candidate_limit(legacy) == 2
    assert result[10][:2] == ['http://a', 'http://b']
    assert len(result[10]) == 4 and set(result[10][2:]) <= {'http://c', 'http://d'}
    assert 'http://e' not in result[10] and 'http://other-owner' not in result[10]
    # Legacy targets keep two alternatives, with an unchanged single owned entry.
    assert len(result[11]) == 2 and set(result[11]) <= {'http://c', 'http://d', 'http://e'}
    owned_legacy = assignments([], [farm_row(1, 'http://a', owner='legacy:z', state='ready'),
                                    farm_row(2, 'http://b', owner='legacy:z', state='ready')],
                               {11: legacy})
    assert owned_legacy[11] == ['http://a']


def test_multi_lease_target_without_contract_accepts_any_farm_and_learns():
    farms = [farm_row(1, 'http://a', sha='c' * 64), farm_row(2, 'http://b', sha='d' * 64)]
    target = dict(model='qwen', run_id='pilot', lease_scope='run', target_leases=2, candidate_limit=16,
                  accepted_compatibility=[], compatibility_sha256=None)
    assert sorted(assignments([], farms, {7: target})[7]) == ['http://a', 'http://b']
    target['accepted_compatibility'] = ['d' * 64]
    assert assignments([], farms, {7: target})[7] == ['http://b']


def test_supervisor_pushes_up_to_candidate_limit_for_multi_lease_targets():
    from tests.tpu_swarm.test_borrowing_supervisor import fixture
    from tpu.swarm.ray_train.borrowing_supervisor import tick
    rows, farms, targets, calls, call = fixture()
    for i in (5, 6, 7):
        rows.append(dict(job_id=20 + i, status='RUNNING', cluster=f'farm-{i}', run_id=f'qwen-farm-{i}'))
        farms[f'farm-{i}'] = dict(models=['qwen'], url=f'http://10.0.0.{i}:24800')
    targets['train-10'].update(lease_scope='run', target_leases=2, candidate_limit=3)
    result = tick(rows, 'farm', [10, 11], call)
    pushed = [t for t in result['targets'] if t['job_id'] == 10][0]
    assert pushed['state'] == 'updated' and len(pushed['urls']) == 3
    assert set(pushed['urls']) <= {'http://10.0.0.1:24800', 'http://10.0.0.5:24800',
                                   'http://10.0.0.6:24800', 'http://10.0.0.7:24800'}
    gemma = [t for t in result['targets'] if t['job_id'] == 11][0]
    assert gemma['urls'] == ['http://10.0.0.2:24800']


# --- Timeouts ----------------------------------------------------------------
def test_remote_only_launcher_disables_every_request_deadline(tmp_path):
    from pathlib import Path
    from tpu.swarm.ray_train.commands import client_environment, trainer_environment
    config = remote_only_config(tmp_path)
    ips = ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4']
    env = trainer_environment(config, Path(tmp_path), Path(tmp_path) / 'run', ips, 0)
    assert env['SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC'] == '0'
    assert env['SKYRL_EXTERNAL_WATCHDOG_ABANDON_SEC'] == '0'
    assert env['SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH'] == '0'
    assert env['TPU_PROCESS_BOUNDS'] == '1,1,4' and env['TPU_PROCESS_ADDRESSES'].count(',') == 3
    client = client_environment(config, Path(tmp_path), ips[0])
    assert client['TTD_SAMPLING_PROGRESS_TIMEOUT'] == '-1'
    assert client['SKYRL_BORROWING_URL'] == 'http://10.0.0.1:%d' % config.ports.inference
    assert client['TTD_SAFE_GRADE_MAX_WORKERS'] == '256'
    # Legacy profiles are untouched.
    legacy = Config.load('tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json')
    legacy_env = trainer_environment(legacy, Path(tmp_path), Path(tmp_path) / 'run', ips, 0)
    assert 'SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH' not in legacy_env
    assert client_environment(legacy, Path(tmp_path), ips[0])['TTD_SAMPLING_PROGRESS_TIMEOUT'] != '-1'


def test_sampling_retry_config_negative_disables_stuck_detection(monkeypatch):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    source = Path('third_party/discover/ttt_discover/rl/train.py').read_text()
    module = ast.parse(source)
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_sampling_retry_config')
    namespace = {'os': __import__('os')}
    import sys
    fake = SimpleNamespace(RetryConfig=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, 'tinker', SimpleNamespace(lib=SimpleNamespace(retry_handler=fake)))
    monkeypatch.setitem(sys.modules, 'tinker.lib', SimpleNamespace(retry_handler=fake))
    monkeypatch.setitem(sys.modules, 'tinker.lib.retry_handler', fake)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'train.py', 'exec'), namespace)
    fn = namespace['_sampling_retry_config']
    monkeypatch.setenv('TTD_SAMPLING_PROGRESS_TIMEOUT', '-1')
    assert vars(fn()) == {'enable_stuck_detection': False}
    monkeypatch.setenv('TTD_SAMPLING_PROGRESS_TIMEOUT', '0')
    assert vars(fn()) == {}
    monkeypatch.setenv('TTD_SAMPLING_PROGRESS_TIMEOUT', '900')
    assert vars(fn()) == {'progress_timeout': 900.0}


def test_watchdog_zero_max_redispatch_is_unlimited():
    from pathlib import Path
    source = Path('skyrl/tinker/dispatch.py').read_text()
    assert 'if self.max_redispatch > 0 and attempts >= self.max_redispatch:' in source
    assert '"unlimited"' in source


def test_multi_borrower_run_deadline_closes_everything(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        borrower.run_deadline = 0
        for member in borrower.members.values():
            member.run_deadline = 0
        await until(lambda: borrower.closed and farm.owners == {})
    multi_case(tmp_path, operation)


V6E8_PILOT = 'tpu/swarm/ray_train/profiles/remote-only-v6e8-east5b-qwen-ac2-pilot-20261005.json'


def test_v6e8_remote_only_trainer_owns_all_eight_chips(tmp_path):
    from pathlib import Path
    from tpu.swarm.ray_train.commands import trainer_environment
    config = Config.load(V6E8_PILOT)
    config.validate()
    assert (config.accelerator, config.hosts, config.trainer.hosts) == ('tpu-v6e-8', 1, 1)
    assert config.trainer.tp * config.trainer.fsdp == 8 and config.single_bucket
    env = trainer_environment(config, Path(tmp_path), Path(tmp_path) / 'run', ['10.0.0.1'], 0)
    assert env['TPU_VISIBLE_CHIPS'] == '0,1,2,3,4,5,6,7'
    assert (env['TPU_PROCESS_BOUNDS'], env['TPU_CHIPS_PER_PROCESS_BOUNDS']) == ('1,1,1', '2,4,1')


@pytest.mark.parametrize('change', [dict(chip_bounds='2,2,1'), dict(tp=4, fsdp=1)])
def test_v6e8_remote_only_trainer_rejects_partial_chip_layouts(change):
    from dataclasses import replace
    config = Config.load(V6E8_PILOT)
    config = replace(config, trainer=replace(config.trainer, **change))
    with pytest.raises(ValueError):
        config.validate()
