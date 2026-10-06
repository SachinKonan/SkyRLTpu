"""Simple deterministic legalizer; no external placer or scoring dependency."""
import time
import numpy as np


def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    deadline = time.monotonic() + time_budget_s
    p = problem
    pos = np.array(p['initial_positions'], dtype=np.float64, copy=True)
    sizes = np.asarray(p['sizes'], dtype=np.float64)
    fixed = np.asarray(p['fixed'], dtype=bool)
    canvas = np.asarray(p['canvas'], dtype=np.float64)
    hard = int(p['num_hard'])
    gap = max(1e-4, float(canvas.max()) * 1e-5)
    lo, hi = sizes / 2 + gap, canvas - sizes / 2 - gap
    if (lo > hi).any():
        raise ValueError('macro cannot fit in canvas with numerical margin')
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])
    occupied = list(np.flatnonzero(fixed[:hard]))
    order = sorted(np.flatnonzero(~fixed[:hard]), key=lambda i: -float(np.prod(sizes[i])))
    for i in order:
        if time.monotonic() > deadline:
            raise TimeoutError('seed legalizer budget exhausted')
        # Candidate centers align with canvas edges and already placed edges.
        # Test nearest candidates first; this is a greedy heuristic, not a
        # guarantee that every geometrically feasible instance can be packed.
        xs = [lo[i, 0], hi[i, 0], pos[i, 0]]
        ys = [lo[i, 1], hi[i, 1], pos[i, 1]]
        for j in occupied:
            delta = (sizes[i] + sizes[j]) / 2 + gap
            xs.extend([pos[j, 0] - delta[0], pos[j, 0] + delta[0]])
            ys.extend([pos[j, 1] - delta[1], pos[j, 1] + delta[1]])
        xs = np.unique(np.clip(xs, lo[i, 0], hi[i, 0]))
        ys = np.unique(np.clip(ys, lo[i, 1], hi[i, 1]))
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        ranking = np.argsort(np.sum((candidates - pos[i]) ** 2, axis=1))
        chosen = None
        for start in range(0, len(ranking), 256):
            if time.monotonic() > deadline:
                raise TimeoutError('seed legalizer budget exhausted')
            batch = candidates[ranking[start:start+256]]
            if occupied:
                delta = np.abs(batch[:, None, :] - pos[occupied][None, :, :])
                overlaps = (delta < (sizes[i] + sizes[occupied])[None, :, :] / 2 + gap/2).all(axis=2)
                legal = ~overlaps.any(axis=1)
            else:
                legal = np.ones(len(batch), dtype=bool)
            if legal.any():
                chosen = batch[np.flatnonzero(legal)[0]]
                break
        if chosen is None:
            raise ValueError('greedy legalizer could not fit a hard macro')
        pos[i] = chosen
        occupied.append(i)
    pos[fixed] = p['initial_positions'][fixed]
    return {'positions': pos.astype(np.float32)}
