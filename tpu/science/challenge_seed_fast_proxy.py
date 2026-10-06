"""Small legal local-search example using the injected Evaluator."""
import time
import numpy as np


def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    deadline = time.monotonic() + max(0.0, time_budget_s - 5.0)
    rng = np.random.default_rng(seed)
    pos = np.array(problem['initial_positions'], dtype=np.float32, copy=True)
    sizes = np.asarray(problem['sizes'], dtype=np.float64)
    canvas = np.asarray(problem['canvas'], dtype=np.float64)
    hard = int(problem['num_hard'])
    movable = np.flatnonzero(~problem['fixed'])
    gap = max(1e-4, float(canvas.max()) * 1e-5)
    best = pos.copy()
    with Evaluator(problem, pos) as ev:
        current = ev.score()['proxy_cost']
        initial_score = current
        # Illustrative search only: extend or replace the proposal strategy.
        for _ in range(256):
            if not len(movable) or time.monotonic() >= deadline:
                break
            i = int(rng.choice(movable))
            xy = np.asarray(pos[i] + rng.normal(size=2) * canvas * .002,
                            dtype=np.float32)
            if np.any(xy - sizes[i]/2 < gap) or np.any(xy + sizes[i]/2 > canvas-gap):
                continue
            if i < hard:
                overlaps = np.all(np.abs(pos[:hard] - xy) < (sizes[:hard] + sizes[i])/2 + gap, axis=1)
                overlaps[i] = False
                if overlaps.any():
                    continue
            trial = ev.apply(i, xy)
            if trial['proxy_cost'] < current:
                ev.commit()
                pos[i] = xy
                current = ev.score()['proxy_cost']
                best = pos.copy()
            else:
                ev.revert()
        # Rebuild the final state; retain the original legal fallback if needed.
        if ev.rebuild()['proxy_cost'] > initial_score:
            best = np.array(problem['initial_positions'], copy=True)
    return {'positions': best}
