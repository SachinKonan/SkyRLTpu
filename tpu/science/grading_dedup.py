"""Process-local, exact-source reuse within a grading batch, never on disk.

Each caller receives its own result copy. A single cancelled rollout cannot
cancel grading still needed by another rollout. Failures are never cached.
"""
import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class _Entry:
    task: asyncio.Task
    waiters: int = 0


class BatchGrader:
    def __init__(self):
        self.batches = OrderedDict()

    async def grade(self, scope, source, evaluate):
        batch = self.batches.setdefault(scope, {})
        self.batches.move_to_end(scope)
        # Retain two recently accessed scopes for pipelined batches. Active
        # older scopes remain until their consumers finish.
        for old_scope in list(self.batches)[:-2]:
            old = self.batches[old_scope]
            if all(entry.task.done() and not entry.waiters for entry in old.values()):
                del self.batches[old_scope]
        entry = batch.get(source)
        reused = entry is not None
        if entry is None:
            entry = _Entry(asyncio.create_task(evaluate()))
            batch[source] = entry
        entry.waiters += 1
        try:
            return deepcopy(await asyncio.shield(entry.task)), reused
        finally:
            entry.waiters -= 1
            if not entry.waiters and not entry.task.done():
                entry.task.cancel()
            if entry.task.cancelled() or (entry.task.done() and entry.task.exception() is not None):
                if batch.get(source) is entry:
                    del batch[source]
            elif not entry.waiters and entry.task.cancelling():
                if batch.get(source) is entry:
                    del batch[source]


async def grade_once(scope, source, evaluate):
    loop = asyncio.get_running_loop()
    if not hasattr(loop, '_science_batch_grader'):
        loop._science_batch_grader = BatchGrader()
    return await loop._science_batch_grader.grade(scope, source, evaluate)
