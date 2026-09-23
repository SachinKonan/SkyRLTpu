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


# --- Borrower request-failure policy ---------------------------------------
import asyncio
import json

import httpx

from tpu.swarm.ray_train.borrowing import Borrower, RemoteRequestRejected
from tests.tpu_swarm.test_inference_borrowing import Farm, config as borrowing_config


class StatusFarm(Farm):
    """Farm fake whose /v1/completions status is programmable per test."""

    def __init__(self):
        super().__init__()
        self.generate_status = None
        self.capabilities = None

    async def __call__(self, request):
        path = request.url.path
        if path == '/v1/completions' and self.generate_status is not None:
            self.requests.append((request.url.host, path))
            return httpx.Response(self.generate_status, json={'detail': 'rejected'})
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
