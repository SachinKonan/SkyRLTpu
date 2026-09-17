"""JAX-assisted macro placement with hard-macro legalization."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax


def _legalize_hard(problem, seed, deadline):
    p = problem
    pos = np.array(p['initial_positions'], dtype=np.float64, copy=True)
    sizes = np.asarray(p['sizes'], dtype=np.float64)
    fixed = np.asarray(p['fixed'], dtype=bool)
    canvas = np.asarray(p['canvas'], dtype=np.float64)
    hard = int(p['num_hard'])
    gap = max(1e-4, float(canvas.max()) * 1e-5)
    lo = sizes / 2 + gap
    hi = canvas - sizes / 2 - gap
    if (lo > hi).any():
        raise ValueError('macro cannot fit')
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])

    occupied = list(np.flatnonzero(fixed[:hard]))
    order = [i for i in np.flatnonzero(~fixed[:hard])]
    # area descending
    order.sort(key=lambda i: -float(np.prod(sizes[i])))
    for i in order:
        if time.monotonic() > deadline:
            raise TimeoutError('legalizer timeout')
        xs = [float(lo[i, 0]), float(hi[i, 0]), float(pos[i, 0])]
        ys = [float(lo[i, 1]), float(hi[i, 1]), float(pos[i, 1])]
        for j in occupied:
            d = (sizes[i] + sizes[j]) / 2 + gap
            xs.extend([float(pos[j, 0] - d[0]), float(pos[j, 0] + d[0])])
            ys.extend([float(pos[j, 1] - d[1]), float(pos[j, 1] + d[1])])
        xs = np.unique(np.clip(xs, lo[i, 0], hi[i, 0]))
        ys = np.unique(np.clip(ys, lo[i, 1], hi[i, 1]))
        xx, yy = np.meshgrid(xs, ys)
        cand = np.column_stack([xx.ravel(), yy.ravel()])
        ranking = np.argsort(np.sum((cand - pos[i]) ** 2, axis=1))
        chosen = None
        for s in range(0, len(ranking), 256):
            if time.monotonic() > deadline:
                raise TimeoutError('legalizer timeout')
            batch = cand[ranking[s:s + 256]]
            if occupied:
                delta = np.abs(batch[:, None, :] - pos[occupied][None, :, :])
                overlap = (delta < (sizes[i] + sizes[occupied])[None, :, :] / 2 + gap / 2).all(axis=2)
                legal = ~overlap.any(axis=1)
            else:
                legal = np.ones(len(batch), dtype=bool)
            if legal.any():
                chosen = batch[np.flatnonzero(legal)[0]]
                break
        if chosen is None:
            raise ValueError('could not place hard macro')
        pos[i] = chosen
        occupied.append(i)
    pos[fixed] = p['initial_positions'][fixed]
    return pos


def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    started = time.monotonic()
    reserve = 5.0
    budget = max(1e-3, time_budget_s - reserve)
    p = problem
    M = int(p['initial_positions'].shape[0])
    H = int(p['num_hard'])
    canvas = np.asarray(p['canvas'], dtype=np.float64)
    sizes = np.asarray(p['sizes'], dtype=np.float64)
    fixed = np.asarray(p['fixed'], dtype=bool)
    init_pos = np.asarray(p['initial_positions'], dtype=np.float64, copy=True)

    # hard legalization on CPU
    legal_deadline = started + max(1.0, budget * 0.3)
    try:
        pos = _legalize_hard(p, seed, legal_deadline)
    except Exception:
        pos = init_pos.copy()
        pos[~fixed] = np.clip(pos[~fixed], sizes[~fixed] / 2 + 1e-4,
                              canvas - sizes[~fixed] / 2 - 1e-4)

    movable_soft = (~fixed) & (np.arange(M) >= H)
    if not np.any(movable_soft):
        # nothing to optimise
        return {"positions": pos.astype(np.float32)}

    # normalised quantities for stable optimisation
    canvas_j = jnp.asarray(canvas, dtype=jnp.float32)
    sizes_n = jnp.asarray(sizes / canvas, dtype=jnp.float32)
    x_n = jnp.asarray(pos / canvas, dtype=jnp.float32)
    ports_n = jnp.asarray(p['port_positions'] / canvas, dtype=jnp.float32)
    offsets_n = jnp.asarray(p['pin_offset'] / canvas, dtype=jnp.float32)
    owners = jnp.asarray(p['pin_owner'], dtype=jnp.int32)
    weights = jnp.asarray(p['net_weights'], dtype=jnp.float32)
    net_off = np.asarray(p['net_offsets'], dtype=np.int64)
    k = int(len(weights))
    net_ids_np = np.concatenate([np.full(int(net_off[i + 1] - net_off[i]), i, dtype=np.int32)
                                for i in range(k)])
    net_ids = jnp.asarray(net_ids_np, dtype=jnp.int32)
    wire_norm = float(p['wirelength_normalizer'])
    wire_norm_j = jnp.asarray(wire_norm, dtype=jnp.float32)

    movable_mask = jnp.asarray(movable_soft, dtype=bool)
    x_fixed = x_n

    # coarse density grid
    n = 20
    xl = jnp.linspace(0.05, 0.95, n)
    yl = jnp.linspace(0.05, 0.95, n)
    gx, gy = jnp.meshgrid(xl, yl, indexing='ij')
    centers = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
    sigma = jnp.asarray(0.055, dtype=jnp.float32)
    area = jnp.prod(sizes_n, axis=1)

    def objective(x):
        combined = jnp.concatenate([x, ports_n], axis=0)
        pins = combined[owners] + offsets_n
        xmax = jax.ops.segment_max(pins, net_ids, k)
        xmin = jax.ops.segment_min(pins, net_ids, k)
        wire = jnp.sum(weights * jnp.sum((xmax - xmin) * canvas_j, axis=1))
        wire_normed = wire / wire_norm_j
        delta = (x[:, None, :] - centers[None, :, :]) / sigma
        exp_term = jnp.exp(-0.5 * jnp.sum(delta * delta, axis=2))
        occ = jnp.sum(exp_term * area[:, None], axis=0)
        dens = jnp.mean((occ / (jnp.mean(occ) + 1e-8)) ** 2)
        return wire_normed + 0.5 * dens

    optimizer = optax.adam(5e-4)
    opt_state = optimizer.init(x_n)

    @jax.jit
    def step(x, state):
        val, grad = jax.value_and_grad(objective)(x)
        grad = jnp.where(movable_mask[:, None], grad, 0.0)
        upd, state = optimizer.update(grad, state, x)
        x_new = optax.apply_updates(x, upd)
        lo = sizes_n / 2 + 1e-5
        hi = 1.0 - sizes_n / 2 - 1e-5
        x_new = jnp.clip(x_new, lo, hi)
        x_new = jnp.where(movable_mask[:, None], x_new, x_fixed)
        return x_new, state, val

    x_cur = x_n
    best_x = x_n
    best_val = float('inf')
    max_iters = 800
    for _ in range(max_iters):
        if time.monotonic() - started > budget:
            break
        x_cur, opt_state, val = step(x_cur, opt_state)
        v = float(val)
        if v < best_val:
            best_val = v
            best_x = x_cur

    pos_final_n = np.array(best_x, dtype=np.float64)
    pos_final = pos_final_n * canvas

    # final legal checks and margins
    gap = max(1e-4, float(canvas.max()) * 1e-5)
    lo = sizes / 2 + gap
    hi = canvas - sizes / 2 - gap
    soft_idx = (~fixed) & (np.arange(M) >= H)
    pos_final[soft_idx] = np.clip(pos_final[soft_idx], lo[soft_idx], hi[soft_idx])
    pos_final[fixed] = init_pos[fixed]

    # hard-hard overlap check
    hard_idx = np.arange(H)
    overlap = False
    for i in range(len(hard_idx)):
        for j in range(i + 1, len(hard_idx)):
            ii = int(hard_idx[i]); jj = int(hard_idx[j])
            dx = abs(pos_final[ii, 0] - pos_final[jj, 0])
            dy = abs(pos_final[ii, 1] - pos_final[jj, 1])
            if dx < (sizes[ii, 0] + sizes[jj, 0]) / 2 - 1e-9 and dy < (sizes[ii, 1] + sizes[jj, 1]) / 2 - 1e-9:
                overlap = True
                break
        if overlap:
            break
    if overlap or not np.all(np.isfinite(pos_final)):
        # fallback to legalised placement
        pos_final = pos

    # ensure inside canvas
    pos_final = np.clip(pos_final, lo, hi)
    return {"positions": pos_final.astype(np.float32)}
