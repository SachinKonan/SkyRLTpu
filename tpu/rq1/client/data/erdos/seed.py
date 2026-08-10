# ---------------------------------------------------------------
#  Erdos C5 optimizer – step‑function construction
# ---------------------------------------------------------------
import numpy as np
import time
from scipy.optimize import minimize

# ------------- 1. Simplex projection ----------------------------
def project_simplex(v, target, low=0.0, high=1.0):
    """
    Project a vector `v` onto the set
          {x | low <= x_i <= high , Σx_i = target}
    This implementation follows the algorithm of Duchi et al. (2008)
    for an efficient O(n) projection.
    """
    v = np.asarray(v, dtype=np.float64)
    if low == high:
        return np.full(v.shape, low)

    lo, hi = -np.max(v) + low, high - np.min(v) + low
    for _ in range(50):
        alpha = (lo + hi) * 0.5
        tmp = v + alpha
        tmp = np.clip(tmp, low, high)
        if tmp.sum() > target:
            hi = alpha
        else:
            lo = alpha
    alpha = (lo + hi) * 0.5
    proj = v + alpha
    proj = np.clip(proj, low, high)
    return proj


# ------------- 2.  C5 evaluation --------------------------------
def eval_c5(h):
    """
    Fast evaluation of the C5 overlap for a discretised h.
    h is an array of length n, representing values on [0,2].
    """
    n = h.size
    dx = 2.0 / n
    return np.max(np.correlate(h, 1.0 - h, mode="full") * dx)


# ------------- 3.  Get initial guess ----------------------------
def get_initial_h(n_points, rng):
    """
    Return an initial vector h of length `n_points`.
    Uses the global `initial_h_values` if present, otherwise a flat
    distribution.  The result is projected onto the simplex and forced
    to be symmetric.
    """
    try:
        h0 = np.array(initial_h_values, dtype=np.float64)
    except NameError:
        h0 = 0.5 * np.ones(n_points)

    if h0.size != n_points:
        old = np.linspace(0, 2, h0.size, endpoint=False)
        new = np.linspace(0, 2, n_points, endpoint=False)
        h0 = np.interp(new, old, h0, left=h0[0], right=h0[-1])

    h0 = np.clip(h0, 0.0, 1.0)
    target_sum = n_points / 2.0
    h0 = project_simplex(h0, target_sum, 0.0, 1.0)
    return (h0 + h0[::-1]) / 2.0


# ------------- 4.  Simulated annealing --------------------------
def anneal(start_h, time_limit):
    """
    Simulated annealing search starting from the vector `start_h`
    for a duration `time_limit` in seconds.  The routine preserves
    symmetry, the simplex constraint and the 0‑1 box.
    """
    n = start_h.size
    best_h = start_h.copy()
    best_val = eval_c5(best_h)

    cur_h = start_h.copy()
    cur_val = best_val

    T = 1.0
    t0 = time.time()
    while time.time() - t0 < time_limit:
        # choose two different indices at random
        i, j = np.random.choice(n, 2, replace=False)

        # admissible delta that keeps the bounds 0/1 for both indices
        del_max = min(1.0 - cur_h[i], cur_h[j])
        del_min = max(-cur_h[i], -(1.0 - cur_h[j]))
        if del_max <= del_min:
            continue
        delta = np.random.uniform(del_min, del_max)

        # propose a new candidate
        new_h = cur_h.copy()
        new_h[i] += delta
        new_h[j] -= delta

        # enforce symmetry and re‑project onto the simplex
        new_h = (new_h + new_h[::-1]) / 2.0
        new_h = project_simplex(new_h, n / 2.0, 0.0, 1.0)

        new_val = eval_c5(new_h)
        dval = new_val - cur_val

        # Metropolis acceptance
        if dval < 0 or np.random.rand() < np.exp(-dval / max(T, 1e-12)):
            cur_h, cur_val = new_h, new_val
            if new_val < best_val:
                best_h, best_val = new_h.copy(), new_val

        # cool down
        T *= 0.9995
    return best_h, best_val


# ------------- 5.  Final refinement by SLSQP -------------------
def slsqp_refine(h, n_points, maxiter=200):
    """
    Refine the vector `h` by a single SLSQP solve on the half domain.
    The result is projected back onto the simplex.
    """
    m = n_points // 2
    target_half = n_points / 4.0

    def obj(a):
        hh = np.concatenate([a, a[::-1]])
        return eval_c5(hh)

    cons = [{'type': 'eq', 'fun': lambda a: np.sum(a) - target_half}]
    bounds = [(0.0, 1.0)] * m

    res = minimize(obj, h[:m], method='SLSQP',
                   constraints=cons, bounds=bounds,
                   options={'ftol': 1e-10,
                            'maxiter': maxiter,
                            'disp': False})
    if res.success:
        refined = np.concatenate([res.x, res.x[::-1]])
        refined = np.clip(refined, 0.0, 1.0)
        refined = project_simplex(refined, n_points / 2.0, 0.0, 1.0)
        return refined
    return h


# ------------- 6.  Public interface --------------------------------
def run(seed=42, budget_s=1000, **kwargs):
    """
    Main routine.
    * `seed`   – random seed for reproducibility
    * `budget_s` – time budget in seconds (default 1000s)
    Returns
        (best_h, best_c5, n_points)
    where `best_h` is a vector in [0,1] (symmetrical),
    `best_c5` is the resulting C5 bound and
    `n_points` is the discretisation resolution.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------
    # 1.  Settings
    # ----------------------------------------------------------------
    n_points = 240                     # use a finer grid
    target_sum = n_points / 2.0
    target_half = n_points / 4.0

    # ----------------------------------------------------------------
    # 2.  Initialise
    # ----------------------------------------------------------------
    best_h = get_initial_h(n_points, rng)
    best_val = eval_c5(best_h)

    # If already good enough, return immediately
    if best_val <= 0.380800001:
        return best_h, best_val, n_points

    start_time = time.time()
    # ----------------------------------------------------------------
    # 3.  Multi‑start annealing
    # ----------------------------------------------------------------
    max_restarts = 7              # how many random starts to try
    restarts = 0
    while (restarts < max_restarts and
           best_val > 0.380800001 and
           time.time() - start_time < budget_s * 0.9):

        # 3.1  Choice of starting vector
        if restarts == 0:
            start_h = best_h
        else:
            half = rng.uniform(0, 1, m := n_points // 2)
            half = project_simplex(half, target_half, 0.0, 1.0)
            start_h = np.concatenate([half, half[::-1]])
            start_h = project_simplex(start_h, target_sum, 0.0, 1.0)

        # 3.2  Allocate time for this restart
        time_left = budget_s - (time.time() - start_time)
        time_per_restart = min(200.0, time_left * 0.6)

        # 3.3  Anneal
        cand_h, cand_val = anneal(start_h, time_per_restart)
        if cand_val < best_val:
            best_val = cand_val
            best_h = cand_h

        restarts += 1

    # ----------------------------------------------------------------
    # 4.  Final annealing on the best vector with remaining time
    # ----------------------------------------------------------------
    time_used = time.time() - start_time
    remaining = max(budget_s - time_used, 0.0)
    if remaining > 3.0:
        best_h, best_val = anneal(best_h, remaining * 0.9)

    # ----------------------------------------------------------------
    # 5.  Optional local refinement by SLSQP
    # ----------------------------------------------------------------
    best_h = slsqp_refine(best_h, n_points)
    best_val = eval_c5(best_h)

    # Ensure final symmetry and simplex
    best_h = (best_h + best_h[::-1]) / 2.0
    best_h = project_simplex(best_h, target_sum, 0.0, 1.0)

    return best_h, best_val, n_points