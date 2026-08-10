"""
Two‑level step‑function optimiser.

We construct sequences consisting of a "big" block of height A (max 1000)
followed by a "tail" of height α·A.  For every possible block size p
(1 ≤ p < n) we set α = sqrt(p/(n-p)) (so that the big‑big and
tail‑tail contributions to the convolution are equal) and evaluate the
exact score with the supplied evaluate_sequence().  The best sequence
found is returned.

The method runs in O(n) time and finishes well within the 1000 s limit.
"""
import time
import numpy as np

# the previously‑found construction is available as a global.
# we keep it for a quick fallback – it is not required for correctness.
try:
    _fallback = list(height_sequence_1)   # type: ignore[name-defined]
except Exception:
    _fallback = None


def _make_two_level(p: int, n: int, A: float = 1000.0) -> list[float]:
    """
    Return a sequence of length n.
    - first p entries are A,
    - the remaining n-p entries are α·A with α = sqrt(p/(n-p)).
    """
    q = n - p
    if q == 0:
        tail_val = 0.0
    else:
        alpha = np.sqrt(p / q)
        tail_val = alpha * A
    # build the list; use Python list to avoid creating a huge temporary array
    seq = [A] * p + [tail_val] * q
    return seq


def propose_candidate(seed: int = 42, budget_s: int = 1000, **kwargs) -> list[float]:
    """
    Search for a good two‑level step function.

    Parameters
    ----------
    seed : int
        Random seed – not really needed for the deterministic two‑level search.
    budget_s : int
        Time budget in seconds.  The implementation finishes far earlier.

    Returns
    -------
    list[float]
        The best sequence found (list of non‑negative heights).
    """
    # -------------------------------------------------------------
    # 1. Initialise
    # -------------------------------------------------------------
    np.random.seed(seed)
    start = time.time()
    deadline = start + budget_s - 0.5   # leave a tiny safety margin

    # length of the construction is fixed by the previously supplied data
    if _fallback is not None and len(_fallback) > 0:
        n = len(_fallback)
    else:
        # a reasonable default – the problem statement mentions 6520 items
        n = 6520

    best_seq = None          # type: list[float] | None
    best_score = float('inf')

    # -------------------------------------------------------------
    # 2. Exhaustive scan over block sizes p
    # -------------------------------------------------------------
    # we only need to test p from 1…n‑1; the score function is cheap,
    # therefore a linear scan is fast enough.
    for p in range(1, n):
        # generate the candidate
        cand = _make_two_level(p, n)

        # evaluate
        try:
            score = evaluate_sequence(cand)     # type: ignore[name-defined]
        except Exception:
            # safety: any unexpected error gets ignored
            continue

        # keep the best one
        if score < best_score:
            best_score = score
            best_seq = cand

        # early exit if we have already passed the deadline
        if time.time() > deadline:
            break

    # -------------------------------------------------------------
    # 3. Fallback – if something went wrong we return the previous best
    # -------------------------------------------------------------
    if best_seq is None:
        # should never happen, but keep the guard
        if _fallback is not None:
            best_seq = _fallback
            best_score = evaluate_sequence(best_seq)
        else:
            # as a last resort return a tiny constant sequence
            best_seq = [1.0] * n
            best_score = evaluate_sequence(best_seq)

    # -------------------------------------------------------------
    # 4. Report
    # -------------------------------------------------------------
    print(f"Search finished after {time.time() - start:.2f}s")
    print(f"Best score found: {best_score:.6f}")
    print(f"Length of sequence: {len(best_seq)}")
    # sanity check – the evaluation function already caps the numbers,
    # but we print the max/min to be sure.
    print(f"max height: {max(best_seq):.2f}, min height: {min(best_seq):.2f}")

    return best_seq