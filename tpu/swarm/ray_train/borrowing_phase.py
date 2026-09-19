"""Client-side sampling boundary; optional and independent of training algorithms."""
from contextlib import asynccontextmanager
import asyncio
import functools
import logging
import os
import uuid

import httpx

log = logging.getLogger(__name__)


@asynccontextmanager
async def sampling_phase(url=None, *, bootstrap=False, expected_n=None):
    url = url or os.environ.get('SKYRL_BORROWING_URL', '')
    if not url:
        yield
        return
    phase = uuid.uuid4().hex
    prepare = float(os.environ.get('SKYRL_BORROWING_PREPARE_TIMEOUT', '130'))
    release = float(os.environ.get('SKYRL_BORROWING_RELEASE_TIMEOUT', '20'))
    interval = float(os.environ.get('SKYRL_BORROWING_HEARTBEAT_SECONDS', '30'))
    expected_n = expected_n if expected_n is not None else int(os.environ.get('GROUP_SIZE', '32'))
    async with httpx.AsyncClient() as client:
        async def keep_alive():
            while True:
                await asyncio.sleep(interval)
                try:
                    response = await client.post(url + '/skyrl/v1/borrowing/heartbeat',
                        json={'phase_id': phase}, timeout=5)
                    response.raise_for_status()
                except httpx.HTTPError:
                    log.warning('Borrowed inference client heartbeat failed; local routing remains available')
        heartbeat = asyncio.create_task(keep_alive())
        try:
            try:
                response = await client.post(url + '/skyrl/v1/borrowing/begin',
                    json={'phase_id': phase, 'bootstrap': bootstrap, 'expected_n': expected_n}, timeout=prepare)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning('External inference preparation unavailable (%s); using local engines', type(exc).__name__)
            yield
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            # A lost begin acknowledgement can still have reserved a farm.
            # Always send matching phase cleanup; never end a newer phase.
            try:
                response = await client.post(url + '/skyrl/v1/borrowing/end',
                    json={'phase_id': phase}, timeout=release)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning('External inference release unconfirmed (%s); server lease expiry will clean up', type(exc).__name__)


def install(ensemble, cfg):
    if not os.environ.get('SKYRL_BORROWING_URL'):
        return
    if not cfg.pipeline_dataflow or cfg.distill_enabled or cfg.num_substeps != 1 or cfg.pooled_multi_lora:
        raise ValueError('external inference borrowing requires the single-adapter pipelined sampling boundary')
    original = ensemble._pipelined_sampling_phase
    if getattr(original, '_borrowing_hook', False):
        return

    @functools.wraps(original)
    async def wrapped(*args, **kwargs):
        async with sampling_phase(expected_n=getattr(cfg, 'group_size', 32)):
            return await original(*args, **kwargs)
    wrapped._borrowing_hook = True
    ensemble._pipelined_sampling_phase = wrapped
