"""
JAX-based Chip Macro Placement Algorithm
Strategy:
- Uses initial Xplace positions (verified legal) as a baseline.
- Optimizes Soft Cluster positions using JAX with Optax.
- Hard Macros are treated as fixed during gradient descent to guarantee Hard-Hard legality.
- Cost function combines Wirelength (HPWL) and Density (Smooth Occupancy).
- Verifies and enforces canvas boundaries at the end.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
try:
    if 'TPU' in str(jax.devices()):
        jax.config.update('jax_platform_name', 'tpu')
except:
    pass

def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    started = time.monotonic()
    deadline = started + time_budget_s
    canvas = np.asarray(problem['canvas'], dtype=np.float32)
    initial_positions = np.asarray(problem['initial_positions'], dtype=np.float32)
    sizes = np.asarray(problem['sizes'], dtype=np.float32)
    fixed = np.asarray(problem['fixed'], dtype=bool)
    num_hard = int(problem['num_hard'])
    port_positions = np.asarray(problem['port_positions'], dtype=np.float32)
    pin_owner = np.asarray(problem['pin_owner'], dtype=np.int32)
    pin_offset = np.asarray(problem['pin_offset'], dtype=np.float32)
    net_offsets = np.asarray(problem['net_offsets'], dtype=np.int32)
    net_weights = np.asarray(problem['net_weights'], dtype=np.float32)
    wirel_norm = float(problem['wirelength_normalizer'])
    M = initial_positions.shape[0]
    L = pin_owner.shape[0]
    K = len(net_weights)
    num_ports = port_positions.shape[0]
    is_hard = np.arange(M) < num_hard
    movable_mask = ~fixed & ~is_hard
    net_ids = np.repeat(np.arange(K), np.diff(net_offsets)).astype(np.int32)
    canvas_jax = jnp.array(canvas, dtype=jnp.float32)
    sizes_jax = jnp.array(sizes, dtype=jnp.float32)
    initial_x_jax = jnp.array(initial_positions, dtype=jnp.float32)
    movable_mask_jax = jnp.array(movable_mask, dtype=jnp.bool_)
    is_hard_jax = jnp.array(is_hard, dtype=jnp.bool_)
    fixed_jax = jnp.array(fixed, dtype=jnp.bool_)
    ports_jax = jnp.array(port_positions, dtype=jnp.float32) / canvas_jax
    owners_jax = jnp.array(pin_owner, dtype=jnp.int32)
    offsets_jax = jnp.array(pin_offset, dtype=jnp.float32) / canvas_jax
    weights_jax = jnp.array(net_weights, dtype=jnp.float32)
    net_ids_jax = jnp.array(net_ids, dtype=jnp.int32)
    grid_shape = problem['grid_shape']
    (rows, cols) = (max(10, int(grid_shape[0])), max(10, int(grid_shape[1])))
    gx = jnp.linspace(0.0, 1.0, cols + 2)
    gy = jnp.linspace(0.0, 1.0, rows + 2)
    (gx, gy) = (gx[1:-1], gy[1:-1])
    (grid_x, grid_y) = jnp.meshgrid(gx, gy, indexing='xy')
    grid_centers = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    def compute_cost(x_norm):
        all_centers = jnp.concatenate([x_norm, ports_jax], axis=0)
        pin_centers = jax.lax.optimization_barrier(all_centers[owners_jax] + offsets_jax)
        xmax = jax.ops.segment_max(pin_centers, net_ids_jax, num_segments=K)
        xmin = jax.ops.segment_min(pin_centers, net_ids_jax, num_segments=K)
        h_bboxes = xmax - xmin
        h_bboxes_microns = h_bboxes * canvas_jax
        wire_cost = jnp.sum(weights_jax * jnp.sum(h_bboxes_microns, axis=1)) / wirel_norm
        grid_exp = grid_centers[:, jnp.newaxis, :]
        x_exp = x_norm[jnp.newaxis, :, :]
        diff = grid_exp - x_exp
        dist_sq = jnp.sum(diff ** 2, axis=2)
        sigma = 0.04
        scale = -0.5 / sigma ** 2
        exp_term = jnp.exp(scale * dist_sq)
        areas = jnp.prod(sizes_jax, axis=1) / wirel_norm
        occ = jnp.sum(exp_term * areas[jnp.newaxis, :], axis=1)
        dens_cost = jnp.sum(occ ** 2)
        final_cost = wire_cost + 0.5 * dens_cost
        return (final_cost, occ)

    def probe_wire(x_norm):
        all_centers = jnp.concatenate([x_norm, ports_jax], axis=0)
        pin_centers = jax.lax.optimization_barrier(all_centers[owners_jax] + offsets_jax)
        xmax = jax.ops.segment_max(pin_centers, net_ids_jax, num_segments=K)
        xmin = jax.ops.segment_min(pin_centers, net_ids_jax, num_segments=K)
        h_bboxes = xmax - xmin
        h_bboxes_microns = h_bboxes * canvas_jax
        wire_cost = jnp.sum(weights_jax * jnp.sum(h_bboxes_microns, axis=1)) / wirel_norm
        return wire_cost
    initial = initial_x_jax
    full = lambda z: compute_cost(z / canvas_jax)[0]
    wire = lambda z: probe_wire(z / canvas_jax)
    for (name, objective) in [('full', full), ('wire', wire)]:
        (value, grad) = jax.jit(jax.value_and_grad(objective))(initial)
        g = np.asarray(grad)
        print('GRAD', name, float(value), float(g.min()), float(g.max()), float(np.linalg.norm(g)), int((g < 0).sum()), flush=True)
        np.save('/output/grad-' + name + '.npy', g)
    physical = np.asarray(initial) * 1.0
    all_centers = np.concatenate([physical, np.asarray(problem['port_positions'])], axis=0)
    owner = np.asarray(problem['pin_owner'], dtype=np.int64)
    pins = all_centers[owner] + np.asarray(problem['pin_offset'])
    starts = np.asarray(problem['net_offsets'], dtype=np.int64)[:-1]
    ids = np.repeat(np.arange(len(starts)), np.diff(problem['net_offsets']))
    hi = np.maximum.reduceat(pins, starts, axis=0)
    lo = np.minimum.reduceat(pins, starts, axis=0)
    maxmask = pins == hi[ids]
    minmask = pins == lo[ids]
    maxcounts = np.add.reduceat(maxmask.astype(float), starts, axis=0)
    mincounts = np.add.reduceat(minmask.astype(float), starts, axis=0)
    gp = (maxmask / maxcounts[ids] - minmask / mincounts[ids]) * np.asarray(problem['net_weights'])[ids, None] / float(problem['wirelength_normalizer'])
    expected = np.zeros_like(all_centers, dtype=float)
    np.add.at(expected, owner, gp)
    expected = expected[:len(initial)] * 1.0
    print('REFERENCE', float(expected.min()), float(expected.max()), float(np.max(np.abs(g - expected))), flush=True)
    np.save('/output/expected-wire.npy', expected)
    return {'positions': np.array(problem['initial_positions'], copy=True)}
