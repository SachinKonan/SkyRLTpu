# TTT-Discover next candidate (paste below SYSTEM_V2 + TASK_V2)

Your previous best candidate is supplied as `candidate.py`. You may inspect only
previous TRAIN and DEVELOPMENT reward logs. You must not read holdout metrics,
private proxy labels, official test inputs, or evaluation implementation.

Think of ONE hypothesis explaining the last failure mode, then implement one
substantive change to candidate.py. Examples: conditional covariance vs global,
noise-corrected scale features, supervised nuisance-invariant latent, different
ensemble component calibration, robust aggregation across unrelated statistics.

Design for broad OOD shifts, not a known synthetic perturbation. Do not tune to
a named proxy family, sample index, or score threshold. Use train/normal val
only for calibration. Restore prior best if the candidate fails, times out or
performs worse on development. Do not select using a final holdout.

Return working code and a one-sentence hypothesis. Keep runtime bounded.
