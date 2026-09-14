"""Real local Ray admission test; no TPU or generated code is executed."""
import os
from pathlib import Path
import tempfile
import time

import pytest


def test_ray_assigns_four_chips_and_fifth_task_waits(tmp_path):
    ray = pytest.importorskip('ray')
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from tpu.science.placement_slots import host_resources, task_resources, chip_lock, assigned_chip
    from tpu.science.placement_ray import PlacementPool

    # Logical memory reservations only: this test allocates no model arrays.
    with tempfile.TemporaryDirectory(prefix='plc-ray-', dir='/tmp') as runtime:
        ray.init(address='local', num_cpus=16, resources=host_resources(range(4)),
                 _memory=64*1024**3, object_store_memory=80*1024**2,
                 include_dashboard=False, _temp_dir=runtime)
        refs = []; pool = None
        try:
            pool = PlacementPool(ray.nodes(), timeout=30)

            @ray.remote(num_cpus=4, memory=16*1024**3, resources=task_resources(), max_retries=0)
            def occupy(name, directory):
                chip = assigned_chip(ray.get_runtime_context().get_accelerator_ids())
                folder = Path(directory)
                with chip_lock(chip, folder/'locks'):
                    started = time.time()
                    (folder/(name+'.started')).write_text(str(started))
                    (folder/(name+'.chip')).write_text(str(chip))
                    deadline = time.monotonic()+60
                    while not (folder/(name+'.release')).exists():
                        if time.monotonic()>deadline:
                            raise TimeoutError('test release missing')
                        time.sleep(.05)
                    return dict(chip=chip, started=started, finished=time.time(),
                                resources=ray.get_runtime_context().get_assigned_resources())

            def submit(name):
                return occupy.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pool.groups[0], placement_group_bundle_index=-1,
                    placement_group_capture_child_tasks=False)).remote(name, str(tmp_path))

            refs = [submit(f'first-{c}') for c in range(4)]
            deadline = time.monotonic()+45
            while len(list(tmp_path.glob('first-*.started'))) != 4:
                if time.monotonic()>deadline:
                    pytest.fail('four slots did not start concurrently')
                time.sleep(.1)
            assert {int(p.read_text()) for p in tmp_path.glob('first-*.chip')} == {0, 1, 2, 3}
            queued = submit('queued')
            refs.append(queued)
            ready, _ = ray.wait([queued], timeout=1)
            assert not ready and not (tmp_path/'queued.started').exists()

            (tmp_path/'first-0.release').touch()
            first = ray.get(refs[0], timeout=15)
            deadline = time.monotonic()+15
            while not (tmp_path/'queued.started').exists():
                if time.monotonic()>deadline:
                    pytest.fail('queued task did not acquire the released slot')
                time.sleep(.1)
            assert float((tmp_path/'queued.started').read_text()) >= first['finished']
            assert int((tmp_path/'queued.chip').read_text()) == first['chip']
            # The other three slots remain occupied while chip zero is reused.
            assert not ray.wait(refs[1:4], num_returns=1, timeout=0)[0]
            for name in ['first-1', 'first-2', 'first-3', 'queued']:
                (tmp_path/(name+'.release')).touch()
            results = ray.get(refs, timeout=15)
            assert max(r['started'] for r in results[:4]) < min(r['finished'] for r in results[:4])
            for result in results:
                assert all(result['resources'].get(k) == v for k,v in task_resources().items())
        finally:
            for name in ['first-0', 'first-1', 'first-2', 'first-3', 'queued']:
                (tmp_path/(name+'.release')).touch()
            for ref in refs:
                ray.cancel(ref, force=True)
            if pool:pool.close()
            ray.shutdown()
