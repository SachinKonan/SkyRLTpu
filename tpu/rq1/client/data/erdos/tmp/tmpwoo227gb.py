import numpy as np

def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")
    
    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")
    
    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")
    
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    
    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)
    
    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    
    dx = 2.0 / n_points
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)
    
    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")
    
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    
    return computed_c5


initial_h_values = np.array([3.0765051341340315e-10, 2.4756253673622956e-10, 7.703081248468928e-10, 1.4382253591561857e-10, 4.387798234022481e-09, 5.49846140439883e-06, 5.6985801834663056e-05, 0.0, 3.9498014850220965e-10, 5.776576334933108e-10, 2.130208756016833e-07, 9.946516127435591e-05, 4.515582755630426e-05, 4.552242029553777e-06, 1.9747032425215652e-05, 1.4558486861481172e-05, 0.0015338278165586135, 0.004223693261213003, 0.001745705351065492, 0.0035476864684774834, 0.006425708947899183, 0.014610036190236441, 0.028502750422352775, 0.07936447221579483, 0.15502092573698267, 0.21672476200506682, 0.25731026404214213, 0.33059596868961455, 0.4268920865642592, 0.5016951524891174, 0.5462014235518288, 0.5378353250557348, 0.4922098631942581, 0.42691082214909953, 0.3572070338962244, 0.34749029755167243, 0.394322219133913, 0.4304129657762149, 0.4580263251697601, 0.5076639177069321, 0.570959649432107, 0.6160056632129884, 0.6380054839687415, 0.6503735485606308, 0.6417155182440186, 0.610681878834211, 0.5527521032870137, 0.5289783017125116, 0.5415292853910533, 0.5292691332867948, 0.47725154877715453, 0.4602664972697869, 0.474321045799773, 0.47798114197771296, 0.4666568012769212, 0.4657110076386583, 0.46649411622520337, 0.4185747355405582, 0.31920473917295117, 0.2704662460954466, 0.25314664373239193, 0.23209303492245825, 0.18532320019560544, 0.23785207653733892, 0.3766148123782262, 0.5276000835988276, 0.669381393843284, 0.7566942155578978, 0.7937124620985272, 0.8021116946874698, 0.7820937048760647, 0.7423255067464966, 0.6799938370531553, 0.6726329434708525, 0.7229487358903633, 0.7964890575929993, 0.8779479118868063, 0.8730433321738523, 0.7686412492767468, 0.6942613463925631, 0.6395265471834839, 0.5788977400287365, 0.5132918158934684, 0.4471627238571298, 0.38178581731362393, 0.4156684202326398, 0.5392036053381006, 0.5854977338402942, 0.5481008216897477, 0.5033279053276386, 0.44998032843716196, 0.504872950089814, 0.6689179861603041, 0.7504497234190954, 0.7508813308593513, 0.7910767207772251, 0.8694884097204435, 0.9260689018381384, 0.9599754482529783, 0.9197946144405237, 0.8003971409703886, 0.7689165646636803, 0.8235745517085583, 0.8690734109390841, 0.9171244745324385, 0.9303639197795983, 0.905069815196799, 0.9191174603506937, 0.9739453762842674, 0.9932481462203523, 0.9754798007180981, 0.9708743749363853, 0.9777347730510136, 0.929806454121132, 0.8301272476195669, 0.8257203357755378, 0.915871105030368, 0.9473363547407184, 0.9229424281056483, 0.9104816156890526, 0.9104816156890526, 0.9229424281056483, 0.9473363547407184, 0.915871105030368, 0.8257203357755378, 0.8301272476195669, 0.929806454121132, 0.9777347730510136, 0.9708743749363853, 0.9754798007180981, 0.9932481462203523, 0.9739453762842674, 0.9191174603506937, 0.905069815196799, 0.9303639197795983, 0.9171244745324385, 0.8690734109390841, 0.8235745517085583, 0.7689165646636803, 0.8003971409703886, 0.9197946144405237, 0.9599754482529783, 0.9260689018381384, 0.8694884097204435, 0.7910767207772251, 0.7508813308593513, 0.7504497234190954, 0.6689179861603041, 0.504872950089814, 0.44998032843716196, 0.5033279053276386, 0.5481008216897477, 0.5854977338402942, 0.5392036053381006, 0.4156684202326398, 0.38178581731362393, 0.4471627238571298, 0.5132918158934684, 0.5788977400287365, 0.6395265471834839, 0.6942613463925631, 0.7686412492767468, 0.8730433321738523, 0.8779479118868063, 0.7964890575929993, 0.7229487358903633, 0.6726329434708525, 0.6799938370531553, 0.7423255067464966, 0.7820937048760647, 0.8021116946874698, 0.7937124620985272, 0.7566942155578978, 0.669381393843284, 0.5276000835988276, 0.3766148123782262, 0.23785207653733892, 0.18532320019560544, 0.23209303492245825, 0.25314664373239193, 0.2704662460954466, 0.31920473917295117, 0.4185747355405582, 0.46649411622520337, 0.4657110076386583, 0.4666568012769212, 0.47798114197771296, 0.474321045799773, 0.4602664972697869, 0.47725154877715453, 0.5292691332867948, 0.5415292853910533, 0.5289783017125116, 0.5527521032870137, 0.610681878834211, 0.6417155182440186, 0.6503735485606308, 0.6380054839687415, 0.6160056632129884, 0.570959649432107, 0.5076639177069321, 0.4580263251697601, 0.4304129657762149, 0.394322219133913, 0.34749029755167243, 0.3572070338962244, 0.42691082214909953, 0.4922098631942581, 0.5378353250557348, 0.5462014235518288, 0.5016951524891174, 0.4268920865642592, 0.33059596868961455, 0.25731026404214213, 0.21672476200506682, 0.15502092573698267, 0.07936447221579483, 0.028502750422352775, 0.014610036190236441, 0.006425708947899183, 0.0035476864684774834, 0.001745705351065492, 0.004223693261213003, 0.0015338278165586135, 1.4558486861481172e-05, 1.9747032425215652e-05, 4.552242029553777e-06, 4.515582755630426e-05, 9.946516127435591e-05, 2.130208756016833e-07, 5.776576334933108e-10, 3.9498014850220965e-10, 0.0, 5.6985801834663056e-05, 5.49846140439883e-06, 4.387798234022481e-09, 1.4382253591561857e-10, 7.703081248468928e-10, 2.4756253673622956e-10, 3.0765051341340315e-10])

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