Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

```python
import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # --- Security Checks ---

    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))
```

A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.
```latex
Starting from a nonnegative step function $f=(a_0,\dots,a_{n-1})$ normalized so that $\sum_j a_j=\sqrt{2n}$, set $M=\|f*f\|_\infty$. Next compute $g_0=(b_0,\dots,b_{n-1})$ by solving a linear program, i.e.\ maximizing $\sum_j b_j$ subject to $b_j\ge0$ and $\|f*g_0\|_\infty\le M$; as is standard, the optimum is attained at an extreme point determined by an active set of binding inequalities, here corresponding to important constraints where the convolution bound $(f*g_0)(x)\le M$ is tight and limiting. Rescale $g_0$ to match the normalization, $g=\frac{\sqrt{2n}}{\sum_j b_j}g_0$, and update $f\leftarrow (1-t)f+t g$ for a small $t>0$. Repeating this step produces a sequence with nonincreasing $\|f*f\|_\infty$, and the iteration is continued until it stabilizes.
```

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have 1000 seconds to run, and after that it has to have returned the best sequence it found. If after 1000 seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

You are iteratively optimizing upper bound.
Here is the last code we ran:
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
Current upper bound (lower is better): 1.941125
Target: 1.503. Current gap: 0.438125. Further improvements will also be generously rewarded.
Length of the construction: 6520

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.