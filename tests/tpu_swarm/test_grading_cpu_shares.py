"""Farm grading dispatch shares the host CPUs between families, so 10-CPU qubit programs are not starved."""
import asyncio
from types import SimpleNamespace
import uuid

from tpu.swarm.ray_train.config import Grading
from tpu.swarm.ray_train.grading_service import GradingService

OWNER = dict(lease_id=None, owner_run='run', token='t')
FAMILIES = {'routing': dict(slots_per_host=16, cpus=10, memory_gib=12, memory_max_gib=12),
            'placement': dict(slots_per_host=80, cpus=2, memory_gib=2, memory_max_gib=2)}


def service():
    config = SimpleNamespace(grading=Grading(), grading_families=FAMILIES, accelerator='tpu-v5p-32', run_id='f')
    grading = GradingService(config, ['10.0.0.1'], warmup=False)
    grading.ready = True
    futures = {'routing': [], 'placement': []}

    def dispatch(task, spec):
        future = asyncio.get_running_loop().create_future()
        futures[task].append(future)
        return future
    grading.dispatch, grading.cancel_ref = dispatch, lambda ref: ref.cancel()
    return grading, futures


def body(task):
    spec = (dict(source='x', routing_suite='full') if task == 'routing' else dict(source='x', case='ibm01'))
    return dict(request_id=uuid.uuid4().hex, task=task, spec=spec)


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def test_a_family_alone_still_fills_its_slots():
    async def run():
        grading, futures = service()
        for _ in range(100):
            await grading.submit(OWNER, body('placement'))
        await settle()
        assert len(futures['placement']) == 80  # 160 CPUs, as before
        await grading.close()
    asyncio.run(run())


def test_qubit_programs_get_their_share_while_circuit_cases_keep_arriving():
    async def run():
        grading, futures = service()
        for _ in range(100):
            await grading.submit(OWNER, body('placement'))
        await settle()
        for _ in range(20):
            await grading.submit(OWNER, body('routing'))
        await settle()
        # Routing gets its 80-CPU share at once (Ray runs it as placement cases finish), not one freed pair at a time.
        assert len(futures['routing']) == 8 and grading.dispatched('placement') == 80
        for index in range(40):
            futures['placement'][index].set_result({'correctness': 1, 'metrics': {}})
        await settle()
        assert grading.dispatched('placement') == 40 and len(futures['placement']) == 80  # no refill past its share
        futures['placement'][40].set_result({'correctness': 1, 'metrics': {}})
        await settle()
        assert grading.dispatched('placement') == 40 and len(futures['placement']) == 81  # refilled within its share
        assert grading.dispatched('routing') * 10 + grading.dispatched('placement') * 2 == 160  # never over the pool
        for future in futures['routing']:
            future.set_result({'correctness': 1, 'metrics': {}})
        await settle()
        # Routing work keeps its share while it has demand; placement refills only what routing leaves.
        assert grading.dispatched('routing') == 8 and grading.dispatched('placement') == 40
        await grading.close()
    asyncio.run(run())
