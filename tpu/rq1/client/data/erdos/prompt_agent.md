You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \max_k \int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: 1000s for your code to run
- **CPUs**: 2 available

## Rules
- Define `run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

You are iteratively optimizing C₅ bound.
Here is the last code we ran:
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
Here is the C₅ bound before and after running the code above (lower is better): 0.381475 -> 0.381439
Target: 0.3808. Current gap: 0.000639. Further improvements will also be generously rewarded.

You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n=240 samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.


## Local testing note
At grading time `initial_h_values` (the current construction) and `evaluate_erdos_solution()` are pre-imported for your program. For local testing, `initial_h_values` is exactly this list:
```json
[3.08e-10, 2.48e-10, 7.7e-10, 1.44e-10, 4.388e-09, 5.498461e-06, 5.6985802e-05, 0.0, 3.95e-10, 5.78e-10, 2.13021e-07, 9.9465161e-05, 4.5155828e-05, 4.552242e-06, 1.9747032e-05, 1.4558487e-05, 0.001533827817, 0.004223693261, 0.001745705351, 0.003547686468, 0.006425708948, 0.01461003619, 0.028502750422, 0.079364472216, 0.155020925737, 0.216724762005, 0.257310264042, 0.33059596869, 0.426892086564, 0.501695152489, 0.546201423552, 0.537835325056, 0.492209863194, 0.426910822149, 0.357207033896, 0.347490297552, 0.394322219134, 0.430412965776, 0.45802632517, 0.507663917707, 0.570959649432, 0.616005663213, 0.638005483969, 0.650373548561, 0.641715518244, 0.610681878834, 0.552752103287, 0.528978301713, 0.541529285391, 0.529269133287, 0.477251548777, 0.46026649727, 0.4743210458, 0.477981141978, 0.466656801277, 0.465711007639, 0.466494116225, 0.418574735541, 0.319204739173, 0.270466246095, 0.253146643732, 0.232093034922, 0.185323200196, 0.237852076537, 0.376614812378, 0.527600083599, 0.669381393843, 0.756694215558, 0.793712462099, 0.802111694687, 0.782093704876, 0.742325506746, 0.679993837053, 0.672632943471, 0.72294873589, 0.796489057593, 0.877947911887, 0.873043332174, 0.768641249277, 0.694261346393, 0.639526547183, 0.578897740029, 0.513291815893, 0.447162723857, 0.381785817314, 0.415668420233, 0.539203605338, 0.58549773384, 0.54810082169, 0.503327905328, 0.449980328437, 0.50487295009, 0.66891798616, 0.750449723419, 0.750881330859, 0.791076720777, 0.86948840972, 0.926068901838, 0.959975448253, 0.919794614441, 0.80039714097, 0.768916564664, 0.823574551709, 0.869073410939, 0.917124474532, 0.93036391978, 0.905069815197, 0.919117460351, 0.973945376284, 0.99324814622, 0.975479800718, 0.970874374936, 0.977734773051, 0.929806454121, 0.83012724762, 0.825720335776, 0.91587110503, 0.947336354741, 0.922942428106, 0.910481615689, 0.910481615689, 0.922942428106, 0.947336354741, 0.91587110503, 0.825720335776, 0.83012724762, 0.929806454121, 0.977734773051, 0.970874374936, 0.975479800718, 0.99324814622, 0.973945376284, 0.919117460351, 0.905069815197, 0.93036391978, 0.917124474532, 0.869073410939, 0.823574551709, 0.768916564664, 0.80039714097, 0.919794614441, 0.959975448253, 0.926068901838, 0.86948840972, 0.791076720777, 0.750881330859, 0.750449723419, 0.66891798616, 0.50487295009, 0.449980328437, 0.503327905328, 0.54810082169, 0.58549773384, 0.539203605338, 0.415668420233, 0.381785817314, 0.447162723857, 0.513291815893, 0.578897740029, 0.639526547183, 0.694261346393, 0.768641249277, 0.873043332174, 0.877947911887, 0.796489057593, 0.72294873589, 0.672632943471, 0.679993837053, 0.742325506746, 0.782093704876, 0.802111694687, 0.793712462099, 0.756694215558, 0.669381393843, 0.527600083599, 0.376614812378, 0.237852076537, 0.185323200196, 0.232093034922, 0.253146643732, 0.270466246095, 0.319204739173, 0.418574735541, 0.466494116225, 0.465711007639, 0.466656801277, 0.477981141978, 0.4743210458, 0.46026649727, 0.477251548777, 0.529269133287, 0.541529285391, 0.528978301713, 0.552752103287, 0.610681878834, 0.641715518244, 0.650373548561, 0.638005483969, 0.616005663213, 0.570959649432, 0.507663917707, 0.45802632517, 0.430412965776, 0.394322219134, 0.347490297552, 0.357207033896, 0.426910822149, 0.492209863194, 0.537835325056, 0.546201423552, 0.501695152489, 0.426892086564, 0.33059596869, 0.257310264042, 0.216724762005, 0.155020925737, 0.079364472216, 0.028502750422, 0.01461003619, 0.006425708948, 0.003547686468, 0.001745705351, 0.004223693261, 0.001533827817, 1.4558487e-05, 1.9747032e-05, 4.552242e-06, 4.5155828e-05, 9.9465161e-05, 2.13021e-07, 5.78e-10, 3.95e-10, 0.0, 5.6985802e-05, 5.498461e-06, 4.388e-09, 1.44e-10, 7.7e-10, 2.48e-10, 3.08e-10]
```