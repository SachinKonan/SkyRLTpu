You are an expert in combinatorial geometry, discrete mathematics, and numerical optimization.
Your task is the planar unit-distance problem: place exactly N = 65536 DISTINCT points in the
Euclidean plane so that the number of unordered pairs of points at Euclidean distance exactly 1 is
as large as possible.

## Scoring
- A pair counts if |d^2 - 1| <= 1e-10 (squared-distance tolerance). Make distances exact to full
  float64 precision; do not rely on the tolerance.
- Validity: exactly 65536 points, all coordinates finite, and every two points at least 1e-3
  apart. An invalid construction scores 0.
- Your score is unit_pairs / N (the regular N-gon baseline scores 1.0; a triangular lattice patch
  scores about 3.0; the goal is to go as far beyond that as possible).
- If your construction naturally uses a different common distance, scale all coordinates so that
  the repeated distance is exactly 1 before returning.

## Rules
- Define `run_construction()` (no arguments) that returns a list of 65536 (x, y) tuples.
- Your function has 1000 seconds to run; return the best construction found within budget.
- You may use numpy, scipy, math. Make all helper functions top level, no closures or lambdas.
- No filesystem or network IO.
- **Print statements**: use print() to log progress; your output will be shown back to you.
- Include a short docstring at the top summarizing your construction/algorithm.

## Background
Known strong constructions come from sections of scaled integer lattices (points with many
representations of a radius as sums of two squares), triangular-lattice patches, and Minkowski
sums / rotated unions of smaller unit-distance graphs (e.g. Moser spindles). The count of the
best known constructions grows superlinearly, n^(1 + c/log log n).

You are iteratively optimizing unit-distance pairs per point.
Here is the last code we ran:
"""Seed: axis-aligned 256x256 unit square lattice (each interior point pairs with
its right and up neighbor at distance exactly 1)."""


def run_construction():
    pts = []
    for i in range(256):
        for j in range(256):
            pts.append((float(i), float(j)))
    return pts

Current unit-distance pairs per point (higher is better): 1.992188
Target: 4.0. Current gap: 2.007812. Further improvements will also be generously rewarded.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different
algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparameters, etc.
Unless you make a meaningful improvement, you will not be rewarded.
