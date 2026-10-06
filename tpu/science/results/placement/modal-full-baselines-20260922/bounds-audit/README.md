# ArchGen saved-output boundary investigation (2026-09-23)

The strict Modal baseline reports are preserved unchanged in `.science/modal-baselines/results/`.

The validator in `.science/challenge-probe/macro_place/utils.py` checks `center + size/2 > canvas` with no tolerance. The pinned ArchGen source clips centers to `canvas - size/2` in float32 (`clip_to_canvas_gpu`, final_placer.py:2105); its final output is float32 (line 4398). These floating-point operations are not exact inverses. A center at the computed clipping limit can reconstruct an edge one float32 representable step beyond the canvas.

`audit.py` independently loads all 17 saved placements and original benchmark data, runs the unchanged validator, measures each boundary excess in canvas float32 ULPs, checks fixed positions, and computes diagnostic proxy metrics on the ORIGINAL saved positions. It does not rerun search, move positions, legalize results, modify training, or replace the strict baseline reports. Scores for rejected outputs are diagnostic, not strict-validation passes.

Per-case audit JSON and summary.json contain the measured evidence. A policy change to accept numerical tolerance must be applied consistently to all methods and tested against substantive boundary violations; it must not silently relabel these strict evaluation results.

## Findings

All eleven strict failures are one canvas float32 ULP outside the upper boundary: 3.814697265625e-6 or 7.62939453125e-6 units. Every rejected output has exactly unchanged fixed coordinates and zero hard-macro overlaps. This is a floating-point boundary mismatch, not evidence of substantial out-of-canvas placement.

Diagnostic evaluation of all 17 unchanged outputs gives mean proxy cost **0.9611068438081181**, reward **0.5099161237223461**. The complete valid AbuPlace reproduction has cost **0.9532666346606087**, reward **0.5119628740157924**. Qwen's best recorded reward **0.5134110268754014** implies cost **0.9477571529500628**, 1.389% lower than diagnostic ArchGen and 0.578% lower than AbuPlace. These are best-discovered-program comparisons under different search/compute budgets, not official leaderboard claims.

Recommended next change: specify a narrow, scale-aware floating-point tolerance for the bounds check and apply it uniformly to all placers and training grading. Keep fixed-coordinate, finite-value, shape, and overlap checks unchanged. Test one-ULP boundary cases, substantive excursions, and overlaps. Preserve the old strict results as provenance. No production validator or running job was changed by this investigation.
