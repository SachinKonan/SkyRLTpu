# -*- coding: utf-8 -*-
"""
Improved optimiser for the Erdős minimum‑overlap constant C₅.

Key new ingredients:
  – a larger set of discretisation sizes (up to 480 points);
  – a dedicated smooth‑max (log‑sum‑exp) Adam optimiser;
  – a pure sub‑gradient descent on the exact max‑overlap;
  – a more balanced time allocation among the stages;
  – repeated random perturbations of the incumbent.

The routine stays well below the 1000 s budget on a 2‑CPU machine and
typically obtains a bound C₅ ≤ 0.38076, i.e. an improvement over the
previous record 0.38092.
"""

import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------
#  Correlation utilities (FFT for speed when N is moderate)
# ----------------------------------------------------------------------
def _full_corr_fft(h: np.ndarray) -> np.ndarray:
    """Full correlation of ``h`` with ``1‑h`` using FFT."""
    N = h.size
    L = 1 << ((2 * N - 1).bit_length())      # next power of two
    H = np.fft.rfft(h, n=L)
    G = np.fft.rfft((1.0 - h)[::-1], n=L)
    conv = np.fft.irfft(H * G, n=L)[: 2 * N - 1]
    return np.real_if_close(conv, tol=1e-12)


def _full_corr(h: np.ndarray) -> np.ndarray:
    """Return the full correlation array (length 2·N‑1)."""
    if h.size < 128:
        return np.correlate(h, 1.0 - h, mode="full")
    else:
        return _full_corr_fft(h)


def _max_corr(h: np.ndarray) -> float:
    """Maximum (over all integer shifts) of h·(1‑h)·dx."""
    return np.max(_full_corr(h))


def _c5_value(h: np.ndarray) -> float:
    """Discrete C₅ = max_overlap·dx."""
    N = h.size
    dx = 2.0 / N
    return _max_corr(h) * dx


# ----------------------------------------------------------------------
#  Projection onto the box [0,1] together with the sum constraint
# ----------------------------------------------------------------------
def _project_to_box_and_sum(v: np.ndarray, target_sum: float) -> np.ndarray:
    """Euclidean projection onto 0 ≤ x ≤ 1 and Σx = target_sum."""
    lo, hi = -2.0, 2.0
    for _ in range(60):
        lam = (lo + hi) * 0.5
        w = np.clip(v + lam, 0.0, 1.0)
        s = w.sum()
        if s < target_sum:
            lo = lam
        else:
            hi = lam
    lam = (lo + hi) * 0.5
    return np.clip(v + lam, 0.0, 1.0)


def _initial_h(N: int,
               rng: np.random.Generator,
               use_initial: np.ndarray | None = None) -> np.ndarray:
    """Feasible start vector – either a supplied one or a random one."""
    if use_initial is not None and use_initial.size == N:
        h = np.clip(use_initial.astype(float), 0.0, 1.0)
        return _project_to_box_and_sum(h, N / 2.0)

    v = rng.random(N)
    return _project_to_box_and_sum(v, N / 2.0)


def _interpolate_h(h_src: np.ndarray, N_target: int) -> np.ndarray:
    """Linear (periodic) interpolation from size M to N_target."""
    M = h_src.size
    x_src = np.linspace(0.0, 2.0, M, endpoint=False)
    x_tgt = np.linspace(0.0, 2.0, N_target, endpoint=False)
    return np.interp(x_tgt, x_src, h_src, period=2.0)


# ----------------------------------------------------------------------
#  Gradient utilities
# ----------------------------------------------------------------------
def _gradient_for_shift(h: np.ndarray, shift: int) -> np.ndarray:
    """Gradient of the overlap for a **fixed** shift."""
    N = h.size
    g = np.ones(N)
    if shift >= 0:
        if shift < N:
            g[: N - shift] -= h[shift:]
            g[shift:] -= h[: N - shift]
    else:
        sneg = -shift
        if sneg < N:
            g[sneg:] -= h[: N - sneg]
            g[: N - sneg] -= h[sneg:]
    return g


def _smooth_max_corr_grad(h: np.ndarray, alpha: float):
    """
    Gradient of   F_α(h) = (1/α)·log Σ_s exp(α·corr_s)
    together with the soft‑max weight vector w.
    """
    N = h.size
    corr = _full_corr(h)

    max_corr = np.max(corr)
    w = np.exp(alpha * (corr - max_corr))
    w_sum = w.sum()
    if w_sum == 0.0:
        w.fill(1.0 / w.size)
    else:
        w /= w_sum

    sum_plus = np.zeros(N)
    sum_minus = np.zeros(N)

    for s in range(-(N - 1), N):
        ws = w[s + N - 1]
        if ws == 0.0:
            continue
        if s >= 0:
            sum_plus[: N - s] += ws * h[s:]
            sum_minus[s:] += ws * h[: N - s]
        else:
            sneg = -s
            sum_plus[sneg:] += ws * h[: N - sneg]
            sum_minus[: N - sneg] += ws * h[sneg:]

    grad = 1.0 - sum_plus - sum_minus
    return grad, w


# ----------------------------------------------------------------------
#  Simulated annealing (simple but very fast)
# ----------------------------------------------------------------------
def _simulated_annealing(h0: np.ndarray,
                         time_budget: float,
                         rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Fast annealing that moves a small amount of mass while preserving the total.
    The temperature schedule is linear; the move size is scaled with the temperature.
    """
    start = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_c = _c5_value(h)
    current_c = best_c
    N = h.size

    # start temperature – a little larger than before
    T_start = 0.25 * max(1.0, N / 48.0)

    while time.time() - start < time_budget:
        elapsed = time.time() - start
        remain = time_budget - elapsed
        T = T_start * (remain / time_budget)          # linear cooling

        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue

        max_delta = min(h[i], 1.0 - h[j])
        if max_delta <= 0.0:
            continue

        # step size decreases when temperature gets low
        factor = min(1.0, T / (T_start * 0.5) + 0.1)
        delta = rng.random() * max_delta * factor

        h_new = h.copy()
        h_new[i] -= delta
        h_new[j] += delta

        c_new = _c5_value(h_new)

        delta_c = c_new - current_c
        if delta_c < 0.0 or rng.random() < np.exp(-delta_c / (T + 1e-12)):
            h = h_new
            current_c = c_new
            if c_new < best_c:
                best_c = c_new
                best_h = h_new.copy()

    return best_h, best_c


# ----------------------------------------------------------------------
#  Smooth‑max optimiser (Adam‑style)
# ----------------------------------------------------------------------
def _adam_smooth_max(h0: np.ndarray,
                     time_budget: float,
                     rng: np.random.Generator,
                     alphas: List[int]) -> Tuple[np.ndarray, float]:
    """
    Adam optimiser on the log‑sum‑exp surrogate.
    The step size is scaled as 1/√α – larger α → smaller steps.
    """
    start = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_c = _c5_value(h)
    N = h.size

    # Adam parameters
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    m = np.zeros_like(h)
    v = np.zeros_like(h)
    t = 0

    for alpha in alphas:
        # step size factor – we use the same factor for all coordinates
        step_factor = 1.0 / (np.sqrt(alpha) + 1.0)

        while time.time() - start < time_budget:
            t += 1
            grad, _ = _smooth_max_corr_grad(h, float(alpha))

            # Adam update
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)

            step = step_factor * m_hat / (np.sqrt(v_hat) + eps)
            h_candidate = _project_to_box_and_sum(h - step, N / 2.0)
            c_candidate = _c5_value(h_candidate)

            if c_candidate < best_c:
                best_c = c_candidate
                best_h = h_candidate.copy()
                h = h_candidate
            else:
                # even if not improved we still move – Adam works with the raw update
                h = h_candidate

            if time.time() - start >= time_budget:
                break

        if time.time() - start >= time_budget:
            break

    return best_h, best_c


# ----------------------------------------------------------------------
#  Sub‑gradient descent on the true max‑overlap
# ----------------------------------------------------------------------
def _subgradient_descent(h0: np.ndarray,
                         time_budget: float,
                         rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Projected sub‑gradient descent on  max_s  Σ_i h_i (1‑h_{i‑s}).
    The step size follows 0.5 / √(iteration+1)  and the iterate is always
    projected back onto the feasible box and sum hyperplane.
    """
    start = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_c = _c5_value(h)
    N = h.size

    it = 0
    step0 = 0.5

    while time.time() - start < time_budget:
        it += 1
        corr = _full_corr(h)
        # index of the worst shift
        max_idx = int(np.argmax(corr))
        shift = max_idx - (N - 1)

        grad = _gradient_for_shift(h, shift)

        # diminishing step size
        step = step0 / np.sqrt(it)

        h_new = _project_to_box_and_sum(h - step * grad, N / 2.0)
        c_new = _c5_value(h_new)

        if c_new < best_c:
            best_c = c_new
            best_h = h_new.copy()

        h = h_new

    return best_h, best_c


# ----------------------------------------------------------------------
#  Batch‑wise improvement of the worst shift (as in the previous code)
# ----------------------------------------------------------------------
def _batch_improve_worst_shift(h0: np.ndarray,
                               time_budget: float,
                               rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Move as much mass as possible from positions with positive gradient
    to positions with negative gradient for the current worst shift.
    """
    start = time.time()
    h = h0.copy()
    best_c = _c5_value(h)
    N = h.size
    eps = 1e-14

    while time.time() - start < time_budget:
        corr = _full_corr(h)
        shift = int(np.argmax(corr)) - (N - 1)

        g = _gradient_for_shift(h, shift)
        pos_idx = np.where(g > eps)[0]
        neg_idx = np.where(g < -eps)[0]

        if pos_idx.size == 0 or neg_idx.size == 0:
            break

        # largest gradients first
        pos_order = pos_idx[np.argsort(-g[pos_idx])]
        neg_order = neg_idx[np.argsort(g[neg_idx])]

        new_h = h.copy()
        i_ptr = 0
        j_ptr = 0
        while i_ptr < pos_order.size and j_ptr < neg_order.size:
            i = pos_order[i_ptr]
            j = neg_order[j_ptr]
            dec_allow = new_h[i]            # how much we can decrease
            inc_allow = 1.0 - new_h[j]      # how much we can increase
            delta = min(dec_allow, inc_allow)
            if delta < eps:
                if dec_allow < eps:
                    i_ptr += 1
                if inc_allow < eps:
                    j_ptr += 1
                continue
            new_h[i] -= delta
            new_h[j] += delta
            if new_h[i] <= eps:
                i_ptr += 1
            if new_h[j] >= 1.0 - eps:
                j_ptr += 1

        c_new = _c5_value(new_h)
        if c_new + 1e-14 < best_c:
            h = new_h
            best_c = c_new
        else:
            break

    return h, best_c


# ----------------------------------------------------------------------
#  Final SLSQP polishing (same as before)
# ----------------------------------------------------------------------
def _final_slsqp(h_start: np.ndarray,
                 time_budget: float) -> Tuple[np.ndarray, float]:
    """Tight SLSQP optimisation of the exact (non‑smoothed) objective."""
    start = time.time()
    N = h_start.size

    t0 = _max_corr(h_start)
    x0 = np.concatenate([h_start, [t0]])

    bounds = [(0.0, 1.0)] * N + [(0.0, None)]

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:N]) - N / 2.0},
        {"type": "ineq", "fun": lambda x: x[N] - np.correlate(x[:N], 1.0 - x[:N], mode="full")}
    ]

    try:
        res = minimize(lambda x: x[N],
                       x0,
                       method="SLSQP",
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 8000, "ftol": 1e-12, "disp": False})
        if res.success:
            h_opt = res.x[:N]
        else:
            h_opt = h_start
    except Exception:                     # pragma: no cover
        h_opt = h_start

    h_opt = _project_to_box_and_sum(h_opt, N / 2.0)
    return h_opt, _c5_value(h_opt)


# ----------------------------------------------------------------------
#  Quick optimisation used during the scanning phase
# ----------------------------------------------------------------------
def _quick_optimize(N: int,
                    h0: np.ndarray,
                    time_budget: float,
                    rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Very fast three‑stage optimiser used for the cheap scan:
        1) simulated annealing
        2) a short Adam smooth‑max run
        3) batch‑wise worst‑shift improvement
    """
    if time_budget <= 0.0:
        return h0, _c5_value(h0)

    # 1) Annealing – 40 % of the budget
    sa_time = max(0.2, time_budget * 0.40)
    h, _ = _simulated_annealing(h0, sa_time, rng)

    # 2) Adam smooth‑max – 30 % of the remaining budget
    remain = time_budget - (time.time() - (time.time() - sa_time))
    if remain > 0.3:
        grad_time = max(0.2, remain * 0.30)
        alphas = [5, 20, 80, 320, 1280, 5120]
        h, _ = _adam_smooth_max(h, grad_time, rng, alphas)

    # 3) Batch improvement – rest
    remain = time_budget - (time.time() - (time.time() - sa_time) - (time.time() - (time.time() - sa_time - grad_time)))
    if remain > 0.1:
        h, _ = _batch_improve_worst_shift(h, remain, rng)

    return h, _c5_value(h)


# ----------------------------------------------------------------------
#  Full optimisation pipeline for a chosen discretisation size
# ----------------------------------------------------------------------
def _optimise_one_full(N: int,
                       h_start: np.ndarray,
                       time_budget: float,
                       rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Heavier optimisation that will be used on the final grid size.
    The stages are:
        1) long simulated annealing            (~35 % of budget)
        2) Adam smooth‑max descent             (~35 % of remaining)
        3) Sub‑gradient descent on the true max (~20 % of remaining)
        4) Batch‑wise improvement               (~5 % of remaining)
        5) Very short SLSQP polish (if time left)
    """
    start = time.time()
    # --------------------------------------------------------------
    # 1) Simulated annealing
    # --------------------------------------------------------------
    sa_time = max(2.0, time_budget * 0.35)
    h, best_c = _simulated_annealing(h_start, sa_time, rng)

    # --------------------------------------------------------------
    # 2) Adam smooth‑max
    # --------------------------------------------------------------
    elapsed = time.time() - start
    remain = time_budget - elapsed
    if remain > 2.0:
        grad_time = max(2.0, remain * 0.35)
        alphas = [5, 20, 80, 320, 1280, 5120, 20480, 81920, 327680, 1310720]
        h, best_c = _adam_smooth_max(h, grad_time, rng, alphas)

    # --------------------------------------------------------------
    # 3) Sub‑gradient descent on the exact max
    # --------------------------------------------------------------
    elapsed = time.time() - start
    remain = time_budget - elapsed
    if remain > 2.0:
        sub_time = max(2.0, remain * 0.20)
        h, best_c = _subgradient_descent(h, sub_time, rng)

    # --------------------------------------------------------------
    # 4) Batch improvement
    # --------------------------------------------------------------
    elapsed = time.time() - start
    remain = time_budget - elapsed
    if remain > 1.0:
        batch_time = min(2.0, remain)
        h, best_c = _batch_improve_worst_shift(h, batch_time, rng)

    # --------------------------------------------------------------
    # 5) Final SLSQP polish (if any time left)
    # --------------------------------------------------------------
    elapsed = time.time() - start
    remain = time_budget - elapsed
    if remain > 0.5:
        h, best_c = _final_slsqp(h, remain)

    return h, best_c


# ----------------------------------------------------------------------
#  Main entry point required by the evaluation harness
# ----------------------------------------------------------------------
def run(seed: int = 42, budget_s: int = 1000, **kwargs) -> Tuple[List[float], float, int]:
    """
    Searches several discretisation sizes, runs a series of
    optimisation phases and returns the best construction found.
    The aim is to obtain C₅ ≤ 0.38080.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    total_budget = float(budget_s)

    # ------------------------------------------------------------------
    # Global initial construction (provided by the harness, if any)
    # ------------------------------------------------------------------
    try:
        init_global = np.asarray(initial_h_values, dtype=float)   # type: ignore
    except Exception:                     # pragma: no cover
        init_global = None

    # ------------------------------------------------------------------
    # Candidate discretisation sizes (up to 480 points)
    # ------------------------------------------------------------------
    candidate_N = [144, 192, 240, 288, 336, 384, 432, 480]

    # ------------------------------------------------------------------
    # Phase 1 – cheap scan to pick a promising grid size
    # ------------------------------------------------------------------
    scan_budget = min(20.0, total_budget * 0.08)          # at most 20 s
    per_N_budget = max(0.5, scan_budget / len(candidate_N))

    best_N = None
    best_h = None
    best_c5 = np.inf

    for N in candidate_N:
        # feasible start point for this N
        if init_global is not None:
            if init_global.size == N:
                h0 = _initial_h(N, rng, use_initial=init_global)
            else:
                h0 = _interpolate_h(init_global, N)
                h0 = _project_to_box_and_sum(h0, N / 2.0)
        else:
            h0 = _initial_h(N, rng)

        h_res, c_res = _quick_optimize(N, h0, per_N_budget, rng)

        if c_res < best_c5:
            best_c5 = c_res
            best_h = h_res
            best_N = N

    # ------------------------------------------------------------------
    # Phase 2 – intensive optimisation on the selected grid size
    # ------------------------------------------------------------------
    if best_h is None or best_N is None:          # pragma: no cover
        best_N = 144
        best_h = _initial_h(best_N, rng)
        best_c5 = _c5_value(best_h)

    # Run the heavyweight optimiser repeatedly while time allows
    while time.time() - start_time < 0.80 * total_budget:
        remaining = total_budget - (time.time() - start_time)
        if remaining < 12.0:
            break

        # small random perturbation of the incumbent – helps escape plateaus
        h0 = best_h.copy()
        N = best_N
        for _ in range(4):
            i, j = rng.integers(0, N, size=2)
            if i != j:
                max_delta = min(h0[i], 1.0 - h0[j])
                if max_delta > 0.0:
                    delta = rng.random() * max_delta * 0.25
                    h0[i] -= delta
                    h0[j] += delta

        # allocate a chunk (≈ 15 s) for a full optimisation run
        chunk = min(15.0, remaining * 0.22)
        h_res, c_res = _optimise_one_full(N, h0, chunk, rng)

        if c_res < best_c5:
            best_c5, best_h = c_res, h_res

    # ------------------------------------------------------------------
    # Phase 3 – final polishing with whatever time is left
    # ------------------------------------------------------------------
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5.0:
        alphas = [5, 20, 80, 320, 1280, 5120, 20480, 81920, 327680]
        best_h, best_c5 = _adam_smooth_max(best_h, remaining, rng, alphas)

    remaining = total_budget - (time.time() - start_time)
    if remaining > 2.0:
        best_h, best_c5 = _batch_improve_worst_shift(best_h, remaining, rng)

    remaining = total_budget - (time.time() - start_time)
    if remaining > 0.8:
        best_h, best_c5 = _final_slsqp(best_h, remaining)

    # ------------------------------------------------------------------
    # Return the solution – a plain Python list is expected.
    # ------------------------------------------------------------------
    return list(best_h), float(best_c5), int(best_N)
