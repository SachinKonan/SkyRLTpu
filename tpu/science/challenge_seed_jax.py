"""Simple deterministic legalizer; no external placer or scoring dependency."""
import time
import numpy as np


def legalize(problem: dict, seed: int, *, time_budget_s: float) -> dict:
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


def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    started = time.monotonic()
    pos = legalize(problem, seed, time_budget_s=min(40, time_budget_s))['positions']
    import jax
    import jax.numpy as jnp
    import optax
    p = problem
    canvas = jnp.asarray(p['canvas'], dtype=jnp.float32)
    ports = jnp.asarray(p['port_positions'], dtype=jnp.float32) / canvas
    owners = jnp.asarray(p['pin_owner'], dtype=jnp.int32)
    offsets = jnp.asarray(p['pin_offset'], dtype=jnp.float32) / canvas
    weights = jnp.asarray(p['net_weights'], dtype=jnp.float32)
    net_ids = jnp.asarray(np.repeat(np.arange(len(weights)), np.diff(p['net_offsets'])), dtype=jnp.int32)
    k = len(weights)
    mask = jnp.asarray((~p['fixed']) & (np.arange(len(pos)) >= p['num_hard']))[:,None]
    sizes = jnp.asarray(p['sizes'], dtype=jnp.float32) / canvas
    fixed_pos = jnp.asarray(pos) / canvas
    # Smooth occupancy at a small public grid; this is a search surrogate.
    gx, gy = jnp.meshgrid(jnp.linspace(.025,.975,20),jnp.linspace(.025,.975,20))
    centers = jnp.stack([gx.ravel(),gy.ravel()],axis=1)
    area = jnp.prod(sizes,axis=1)
    def objective(x):
        pins = jnp.concatenate([x, ports])[owners] + offsets
        xmax = jax.ops.segment_max(pins, net_ids, k)
        xmin = jax.ops.segment_min(pins, net_ids, k)
        wire = jnp.sum(weights * jnp.sum((xmax-xmin)*canvas,axis=1)) / p['wirelength_normalizer']
        delta = (x[:,None,:]-centers[None,:,:]) / .055
        occupancy = jnp.sum(jnp.exp(-.5*jnp.sum(delta*delta,axis=2))*area[:,None],axis=0)
        density = jnp.mean((occupancy/(jnp.mean(occupancy)+1e-8))**2)
        return wire + .12*density
    optimizer=optax.adam(.0005)
    @jax.jit
    def step(x,state):
        value,grad = jax.value_and_grad(objective)(x)
        updates,state=optimizer.update(jnp.where(mask,grad,0),state,x)
        x=optax.apply_updates(x,updates)
        x=jnp.where(mask,jnp.clip(x,sizes/2+1e-5,1-sizes/2-1e-5),fixed_pos)
        return x,state,value
    x=fixed_pos;state=optimizer.init(x);best=pos.copy();best_value=float('inf')
    # Synchronize each iteration so the timer includes completed device work.
    for _ in range(400):
        if time.monotonic()-started > time_budget_s-15:break
        previous=x
        x,state,value=step(x,state)
        score=float(value)
        if np.isfinite(score) and score<best_value:
            best_value=score;best=np.array(previous*canvas,dtype=np.float32,copy=True)
    best[:p['num_hard']]=pos[:p['num_hard']]
    best[p['fixed']]=p['initial_positions'][p['fixed']]
    # Restore a generous physical margin after float32 accelerator arithmetic.
    gap=max(1e-4,float(np.max(p['canvas']))*1e-5)
    soft=(~p['fixed']) & (np.arange(len(pos))>=p['num_hard'])
    best[soft]=np.clip(best[soft],p['sizes'][soft]/2+gap,p['canvas']-p['sizes'][soft]/2-gap)
    return {'positions':best}
