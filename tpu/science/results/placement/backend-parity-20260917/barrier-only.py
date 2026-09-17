"""Macro placement with hard legalization and JAX soft optimization."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax


def _legalize_hard(problem, pos, deadline):
    sizes = np.asarray(problem['sizes'], dtype=np.float64)
    fixed = np.asarray(problem['fixed'], dtype=bool)
    canvas = np.asarray(problem['canvas'], dtype=np.float64)
    num_hard = int(problem['num_hard'])
    gap = max(1e-4, float(canvas.max()) * 1e-5)
    lo = sizes / 2 + gap
    hi = canvas - sizes / 2 - gap
    if (lo > hi).any():
        raise ValueError('macro cannot fit')
    pos = pos.copy()
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])
    occupied = list(np.flatnonzero(fixed[:num_hard]))
    order = sorted(np.flatnonzero(~fixed[:num_hard]),
                   key=lambda i: -float(np.prod(sizes[i])))
    for i in order:
        if time.monotonic() > deadline:
            raise TimeoutError('legalizer timeout')
        xs = [float(lo[i, 0]), float(hi[i, 0]), float(pos[i, 0])]
        ys = [float(lo[i, 1]), float(hi[i, 1]), float(pos[i, 1])]
        for j in occupied:
            delta = (sizes[i] + sizes[j]) / 2 + gap
            xs.extend([float(pos[j, 0] - delta[0]), float(pos[j, 0] + delta[0])])
            ys.extend([float(pos[j, 1] - delta[1]), float(pos[j, 1] + delta[1])])
        xs = np.unique(np.clip(xs, lo[i, 0], hi[i, 0]))
        ys = np.unique(np.clip(ys, lo[i, 1], hi[i, 1]))
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        ranking = np.argsort(np.sum((candidates - pos[i]) ** 2, axis=1))
        chosen = None
        for start in range(0, len(ranking), 256):
            if time.monotonic() > deadline:
                raise TimeoutError('legalizer timeout')
            batch = candidates[ranking[start:start + 256]]
            if occupied:
                d = np.abs(batch[:, None, :] - pos[occupied][None, :, :])
                overlap = (d < (sizes[i] + sizes[occupied])[None, :, :] / 2 + gap / 2).all(axis=2)
                legal = ~overlap.any(axis=1)
            else:
                legal = np.ones(len(batch), dtype=bool)
            if legal.any():
                chosen = batch[np.flatnonzero(legal)[0]]
                break
        if chosen is None:
            raise ValueError('hard macro cannot be placed')
        pos[i] = chosen
        occupied.append(i)
    pos[fixed] = problem['initial_positions'][fixed]
    return pos


def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    start = time.monotonic()
    deadline = start + time_budget_s
    p = problem
    M = int(p['sizes'].shape[0])
    canvas = np.asarray(p['canvas'], dtype=np.float64)
    sizes = np.asarray(p['sizes'], dtype=np.float64)
    fixed = np.asarray(p['fixed'], dtype=bool)
    num_hard = int(p['num_hard'])
    init_pos = np.asarray(p['initial_positions'], dtype=np.float64, copy=True)

    # --- hard legalization fallback ---
    pos = init_pos.copy()
    try:
        leg_deadline = start + max(1.0, 0.3 * time_budget_s)
        pos = _legalize_hard(p, pos, leg_deadline)
    except Exception:
        pos = init_pos.copy()

    gap = max(1e-4, float(canvas.max()) * 1e-5)
    lo = sizes / 2 + gap
    hi = canvas - sizes / 2 - gap
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])

    # --- JAX soft optimization ---
    # if no movable soft clusters, return
    movable_soft = (~fixed) & (np.arange(M) >= num_hard)
    if not np.any(movable_soft):
        best_phys = pos.astype(np.float32)
        best_phys[fixed] = p['initial_positions'][fixed]
        return {"positions": best_phys}

    canvas_j = jnp.asarray(canvas, dtype=jnp.float32)
    ports = jnp.asarray(p['port_positions'], dtype=jnp.float32) / canvas_j
    owners = jnp.asarray(p['pin_owner'], dtype=jnp.int32)
    offsets = jnp.asarray(p['pin_offset'], dtype=jnp.float32) / canvas_j
    weights = jnp.asarray(p['net_weights'], dtype=jnp.float32)
    net_offsets = np.asarray(p['net_offsets'], dtype=np.int64)
    net_ids_np = np.repeat(np.arange(len(weights)), np.diff(net_offsets))
    net_ids = jnp.asarray(net_ids_np, dtype=jnp.int32)
    k = len(weights)

    sizes_norm = jnp.asarray(sizes, dtype=jnp.float32) / canvas_j
    mask = jnp.asarray(movable_soft[:, None], dtype=jnp.bool_)
    fixed_pos_norm = jnp.asarray(pos, dtype=jnp.float32) / canvas_j

    # density grid
    grid_n = 24
    xs = jnp.linspace(0.025, 0.975, grid_n)
    ys = jnp.linspace(0.025, 0.975, grid_n)
    gx, gy = jnp.meshgrid(xs, ys)
    centers = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
    area = jnp.prod(sizes_norm, axis=1)
    sigma = 0.055

    def objective(x):
        all_c = jnp.concatenate([x, ports])
        pins = jax.lax.optimization_barrier(all_c[owners] + offsets)
        xmax = jax.ops.segment_max(pins, net_ids, k)
        xmin = jax.ops.segment_min(pins, net_ids, k)
        wl = jnp.sum(weights * jnp.sum((xmax - xmin) * canvas_j, axis=1))
        wire_norm = wl / float(p['wirelength_normalizer'])
        delta = (x[:, None, :] - centers[None, :, :]) / sigma
        occ = jnp.sum(jnp.exp(-0.5 * jnp.sum(delta * delta, axis=2)) * area[:, None], axis=0)
        mean_occ = jnp.mean(occ) + 1e-8
        density = jnp.mean((occ / mean_occ) ** 2)
        return wire_norm + 0.12 * density

    optimizer = optax.adam(5e-4)
    @jax.jit
    def step(x, state):
        val, grad = jax.value_and_grad(objective)(x)
        grad_masked = jnp.where(mask, grad, 0.0)
        updates, new_state = optimizer.update(grad_masked, state)
        x_new = optax.apply_updates(x, updates)
        x_new = jnp.where(mask,
                          jnp.clip(x_new, sizes_norm / 2 + 1e-5, 1.0 - sizes_norm / 2 - 1e-5),
                          fixed_pos_norm)
        return x_new, new_state, val

    x = fixed_pos_norm
    state = optimizer.init(x)
    # warm up compilation
    x, state, _ = step(x, state)

    best_val = float('inf')
    best_phys = pos.copy()
    it = 0
    while time.monotonic() < deadline - 2.0 and it < 800:
        x, state, val = step(x, state)
        it += 1
        v = float(val)
        if v < best_val:
            best_val = v
            best_phys = np.array(x * canvas_j, dtype=np.float64)

    best_phys[:num_hard] = pos[:num_hard]
    best_phys[fixed] = p['initial_positions'][fixed]
    soft_idx = (~fixed) & (np.arange(M) >= num_hard)
    best_phys[soft_idx] = np.clip(best_phys[soft_idx],
                                 lo[soft_idx], hi[soft_idx])

    # final sanity
    if not np.all(np.isfinite(best_phys)):
        best_phys = init_pos.copy()
    best_phys[fixed] = p['initial_positions'][fixed]

    return {"positions": best_phys.astype(np.float32)}
