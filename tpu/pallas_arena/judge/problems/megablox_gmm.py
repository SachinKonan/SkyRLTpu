"""Task 3 — Megablox grouped matmul (MoE), uniform + Zipf-skewed groups.

Baseline: the tuned Pallas gmm that MaxText vendors
(jax.experimental.pallas.ops.tpu.megablox.gmm at our jax pin — same kernel
family; the TPU judge binds it directly). `jax.lax.ragged_dot` is the legal
floor and the CPU stand-in baseline for the test battery. `group_sizes` are
freshly sampled per grading from BOTH uniform and Zipf-skewed distributions
so imbalance-tuned kernels can't fake wins.

kernel(lhs, rhs, group_sizes) -> out
  lhs:         [m, k]              bfloat16 (tokens, grouped contiguously)
  rhs:         [num_groups, k, n]  bfloat16 (per-expert weights)
  group_sizes: [num_groups]        int32, sum == m (zeros allowed!)
  out:         [m, n]              float32, out[rows of group g] = lhs @ rhs[g]
fp32 accumulation. Contract: fwd + bwd (d/d lhs, d/d rhs; `group_sizes` is an
integer routing vector and is not differentiated). MoE training needs gradients
into the expert weights, and megablox ships the backward for it -- `gmm` is a
`jax.custom_vjp` whose `_gmm_bwd` computes the weight gradient with a TRANSPOSED
grouped matmul (`tgmm`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge.problems.base import (
    AdversarialCase,
    Problem,
    ShapeCase,
)


def gmm_reference(lhs, rhs, group_sizes):
    """fp32 closed form via lax.ragged_dot (the design floor) — itself
    cross-checked against a python loop in the CPU battery."""
    return jax.lax.ragged_dot(lhs.astype(jnp.float32), rhs.astype(jnp.float32), group_sizes)


def gmm_loop_reference(lhs, rhs, group_sizes):
    """Independent numpy loop formulation (test-battery cross-check only)."""
    lhs = np.asarray(lhs, np.float32)
    rhs = np.asarray(rhs, np.float32)
    sizes = np.asarray(group_sizes)
    out = np.zeros((lhs.shape[0], rhs.shape[2]), np.float32)
    row = 0
    for g, size in enumerate(sizes):
        if size:
            out[row : row + size] = lhs[row : row + size] @ rhs[g]
        row += size
    return out


def _honest_faithful_bf16(lhs, rhs, group_sizes):
    """bf16 operands into the MXU, fp32 accumulator -- the production path.

    NOTE: on CPU this is indistinguishable from the fp32 reference, because
    ``preferred_element_type`` is a no-op inside a fused op there; it only
    separates from ``reference_bf16`` on a real TPU. It is included so the TPU
    calibration is right, and the CPU battery simply sees it as exact.
    """
    return jax.lax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type=jnp.float32)


def _honest_group_loop_bf16(lhs, rhs, group_sizes):
    """Same math with a per-group reduction ORDER (one dot per expert instead
    of one fused ragged dot), fp32 accumulators. A legal candidate strategy, so
    it must not be punished for a different-but-valid accumulation order."""
    m = lhs.shape[0]
    n = rhs.shape[2]
    rows = jnp.arange(m)
    starts = jnp.concatenate([jnp.zeros((1,), group_sizes.dtype), jnp.cumsum(group_sizes)[:-1]])

    def body(acc, g):
        lo = starts[g]
        hi = lo + group_sizes[g]
        sel = (rows >= lo) & (rows < hi)
        part = jnp.dot(lhs, rhs[g], preferred_element_type=jnp.float32)
        return acc + jnp.where(sel[:, None], part, 0.0), None

    acc, _ = jax.lax.scan(body, jnp.zeros((m, n), jnp.float32), jnp.arange(rhs.shape[0]))
    return acc


def _megablox_tiling(m: int, k: int, n: int):
    """Shape-aware (tile_m, tile_k, tile_n) for the megablox baseline.

    `gmm` accepts a callable exactly so callers can supply a tuned choice; its
    default (128, 128, 128) is a placeholder, not a recommendation. Measured
    best-of-grid on v6e-1 (job 3691513) across tokamax's canonical specs and
    ours: (1024, 1024, 1024) won 4 of 7 shapes, (512, 1024, 1024) won 2, and
    (256, 1024, 2048) won 1 -- so 1024^3 clamped to the real dims is the right
    default, and every tile must divide into its dimension or Mosaic rejects it.

    KNOWN LIMIT: at our declared production `32k-e8` this still lost to XLA
    (1.21x), so the 6-point grid behind this heuristic does not contain the
    optimum everywhere. A boot-time search (the judge owns its chip and already
    measures a noise floor at boot) would do better and would match tokamax's
    own discipline, where an autotuning cache miss is a hard error rather than a
    silent fallback. Tracked separately; this heuristic is 13-55x better than
    what the arena was doing and does not pretend to be optimal.
    """

    def fit(want: int, dim: int) -> int:
        t = min(want, dim)
        while t > 128 and dim % t:
            t //= 2
        return max(128, t) if dim >= 128 else dim

    return (fit(1024, m), fit(1024, k), fit(1024, n))


def _sample_group_sizes(key, num_groups, m, dist):
    # The two DEGENERATE distributions are tokamax's own benchmark corners
    # (ragged_dot/arg_specs.py SPEC_SHAPES): `compute_bound` routes every token
    # to one expert, `memory_bound` gives each exactly one. Uniform/zipf
    # sampling never reaches either, and both are real MoE states -- routing
    # collapse, and a batch smaller than the expert count.
    if dist == "one-hot":
        return jnp.zeros((num_groups,), jnp.int32).at[0].set(m).astype(jnp.int32)
    if dist == "one-each":
        base = m // num_groups
        counts = jnp.full((num_groups,), base, jnp.int32)
        return counts.at[0].add(m - base * num_groups).astype(jnp.int32)
    if dist == "uniform":
        w = jnp.ones((num_groups,))
    elif dist == "zipf":
        w = 1.0 / jnp.arange(1, num_groups + 1, dtype=jnp.float32)
        w = jax.random.permutation(key, w)
    else:
        raise ValueError(dist)
    # multinomial split of m tokens by weights
    probs = w / jnp.sum(w)
    counts = jnp.floor(probs * m).astype(jnp.int32)
    # hand the remainder to the heaviest group (keeps sum == m, determinism)
    rem = m - jnp.sum(counts)
    counts = counts.at[jnp.argmax(probs)].add(rem)
    return counts.astype(jnp.int32)


class MegabloxGmmProblem(Problem):
    name = "megablox_gmm"
    version = "1"
    # BACKWARD IS PART OF THE CONTRACT. A forward-only grouped matmul cannot
    # train a mixture of experts -- the whole point of the op is the expert
    # projections, and those need gradients flowing to the expert weights.
    #
    # This was False, which was simply wrong. VERIFIED in the arena's own jax
    # pin (jax/experimental/pallas/ops/tpu/megablox/ops.py):
    #
    #     line  22:  gmm = jax.custom_vjp(
    #     line  28:  def _gmm_fwd(
    #     line  63:  def _gmm_bwd(
    #     line  90:      grad_rhs = backend.tgmm(
    #
    # so megablox ships a hand-written backward whose weight gradient is a
    # TRANSPOSED grouped matmul (`tgmm`, exported from megablox.backend rather
    # than the package root, which is why a top-level export check misses it).
    # tokamax tests the same thing (`test_vjp`, `test_tgmm_drhs_pipes`).
    #
    # The denominator therefore costs nothing to obtain: because `gmm` is a
    # custom_vjp, differentiating the EXISTING baseline dispatches straight into
    # `_gmm_bwd`/`tgmm`, so the backward is timed against the production
    # backward rather than against autodiff-through-our-reference. That
    # distinction is the whole reason this is safe to turn on -- inventing a
    # weak backward bar is the mistuned-denominator mistake that invalidated
    # every reward this arena produced before the per-shape election.
    has_bwd = True
    require_pallas = True
    general_mode = True  # score the holdout; denominator = fastest honest impl per shape
    memory_bound = False

    def shape_cases(self):
        return [
            # (tokens=32k, experts=8 and 64, k=4096, n=14336)
            ShapeCase("32k-e8-uniform", {"m": 32768, "g": 8, "k": 4096, "n": 14336, "dist": "uniform"}),
            ShapeCase("32k-e8-zipf", {"m": 32768, "g": 8, "k": 4096, "n": 14336, "dist": "zipf"}),
            ShapeCase("32k-e64-uniform", {"m": 32768, "g": 64, "k": 4096, "n": 14336, "dist": "uniform"}),
            ShapeCase("32k-e64-zipf", {"m": 32768, "g": 64, "k": 4096, "n": 14336, "dist": "zipf"}),
            ShapeCase(
                "holdout-16k-e32-zipf", {"m": 16384, "g": 32, "k": 2048, "n": 7168, "dist": "zipf"}, holdout=True
            ),
            # PROBE set: one-chip sizes at the UNCHANGED expert widths
            # (k=4096, n=14336). Only the token count and the expert COUNT
            # shrink -- 4 experts instead of 8/64 -- because `rhs` is the whole
            # memory budget here: [g, 4096, 14336] bf16 is 940 MB at g=8, and
            # the worker holds `#scored x correctness_seeds` full input tuples
            # live at once. At g=4 that is 470 MB a fixture. Zipf is kept so
            # the imbalance (and the zero-size-group trap) survives the shrink,
            # and the holdout m=3000 is deliberately not a multiple of any
            # reasonable row tile.
            ShapeCase("probe-m4096-e4-uniform", {"m": 4096, "g": 4, "k": 4096, "n": 14336, "dist": "uniform"}, probe=True),
            ShapeCase("probe-m2048-e4-zipf", {"m": 2048, "g": 4, "k": 4096, "n": 14336, "dist": "zipf"}, probe=True),
            # GENERAL sweep: tokens and expert COUNT both vary, and the 8x7b
            # orientation (k=14336, n=4096 -- the transpose of ours) is the one
            # tokamax actually benchmarks, where tuned megablox beats XLA 4.1x.
            ShapeCase("probe-m8192-e8-uniform", {"m": 8192, "g": 8, "k": 4096, "n": 4096, "dist": "uniform"}, probe=True),
            # PROVENANCE, tokamax naming: this IS their `8x7b` spec
            # (mixtral, g=8 m=8192 k=14336 n=4096) rather than a shape we
            # invented. Named so a result can be quoted against the same
            # configuration they publish numbers for.
            ShapeCase("mixtral-8x7b-g8-m8192", {"m": 8192, "g": 8, "k": 14336, "n": 4096, "dist": "zipf"}, probe=True),
            # tokamax `compute_bound`: ONE expert takes every token, seven get
            # zero -- group_sizes [4096] + [0]*7. A real MoE routing collapse,
            # and a correctness trap our uniform/zipf sampling never produces.
            ShapeCase("tokamax-compute-bound-g8", {"m": 4096, "g": 8, "k": 4096, "n": 4096, "dist": "one-hot"}, probe=True),
            # tokamax `memory_bound`: one token per expert, [1]*8.
            ShapeCase("tokamax-memory-bound-g8", {"m": 8, "g": 8, "k": 4096, "n": 4096, "dist": "one-each"}, probe=True),
            ShapeCase("probe-m4096-e16-zipf", {"m": 4096, "g": 16, "k": 4096, "n": 4096, "dist": "zipf"}, probe=True),
            # TENSOR PARALLEL (v6e-8): the output feature axis n sharded 8
            # ways -- the classic MoE up-projection split. Per shard n=512.
            ShapeCase("tp8-m4096-e8-uniform", {"m": 4096, "g": 8, "k": 4096, "n": 4096, "dist": "uniform"}, probe=True, tp=8),
            ShapeCase("tp8-holdout-m3000-e8-zipf", {"m": 3000, "g": 8, "k": 4096, "n": 4096, "dist": "zipf"}, probe=True, tp=8, holdout=True),
            ShapeCase("tp4-m4096-e8-uniform", {"m": 4096, "g": 8, "k": 4096, "n": 4096, "dist": "uniform"}, probe=True, tp=4),
            ShapeCase("tp4-holdout-m3000-e8-zipf", {"m": 3000, "g": 8, "k": 4096, "n": 4096, "dist": "zipf"}, probe=True, tp=4, holdout=True),
            ShapeCase(
                "probe-holdout-m3000-e4-zipf",
                {"m": 3000, "g": 4, "k": 4096, "n": 14336, "dist": "zipf"},
                holdout=True,
                probe=True,
            ),
            # CPU battery
            ShapeCase("tiny", {"m": 64, "g": 4, "k": 16, "n": 24, "dist": "uniform"}, smoke=True),
            ShapeCase("tiny-zipf", {"m": 64, "g": 4, "k": 16, "n": 24, "dist": "zipf"}, smoke=True),
            ShapeCase("tiny-holdout", {"m": 32, "g": 4, "k": 8, "n": 16, "dist": "zipf"}, smoke=True, holdout=True),
        ]

    def make_inputs(self, key, case):
        kl, kr, kg = jax.random.split(key, 3)
        dm = case.dims
        lhs = jax.random.normal(kl, (dm["m"], dm["k"]), jnp.float32).astype(jnp.bfloat16)
        rhs = (jax.random.normal(kr, (dm["g"], dm["k"], dm["n"]), jnp.float32) / np.sqrt(dm["k"])).astype(jnp.bfloat16)
        sizes = _sample_group_sizes(kg, dm["g"], dm["m"], dm["dist"])
        return (lhs, rhs, sizes)

    def reference(self, lhs, rhs, group_sizes):
        return gmm_reference(lhs, rhs, group_sizes)

    def grad_outputs(self, kernel_fn, lhs, rhs, group_sizes):
        """d/d(lhs, rhs) of a fixed scalar functional of the output.

        argnums=(0, 1) only: `group_sizes` is an integer routing vector, not a
        differentiable input -- megablox's own `_gmm_bwd` likewise returns no
        cotangent for it.

        The cotangent is a deterministic NON-symmetric probe (cos of a flat
        iota) rather than a plain sum. Summing the output makes a wrong backward
        look right, because errors cancel across the reduction -- and the two
        gradients here have very different structure (`grad_lhs` is a grouped
        matmul against rhs^T, `grad_rhs` is the transposed grouped matmul), so a
        candidate that gets one right and the other wrong must not average out.
        """
        out_shape = (lhs.shape[0], rhs.shape[-1])
        probe = jnp.cos(
            jnp.arange(int(np.prod(out_shape)), dtype=jnp.float32)
        ).reshape(out_shape)

        def scalar(l32, r32):
            out = kernel_fn(l32.astype(lhs.dtype), r32.astype(rhs.dtype), group_sizes)
            return jnp.sum(out.astype(jnp.float32) * probe)

        return jax.grad(scalar, argnums=(0, 1))(
            lhs.astype(jnp.float32), rhs.astype(jnp.float32)
        )

    # Which baseline the last `baseline()` call actually used. Recorded (not
    # asserted) so a boot report says plainly what the score denominator IS.
    baseline_impl: str = "?"

    def baseline(self, lhs, rhs, group_sizes):
        """The tuned Pallas megablox gmm, with `lax.ragged_dot` as the honest
        fallback (the design floor, and the CPU stand-in).

        megablox has its own tiling constraints and refuses some (m, k, n); a
        judge that dies at BOOT on that has graded nothing at all. Which one
        ran is recorded, because a score against `ragged_dot` is not a score
        against the production kernel.

        The `tiling=` is LOAD-BEARING, and this docstring claimed "tuned" for a
        long time while passing none. Measured on v6e-1 (job 3691513), the
        library default (128, 128, 128) costs 13x to 55x against a tuned choice
        -- at our own probe shape, 38.2x. Every megablox verdict this arena has
        recorded was therefore scored against a baseline we had crippled, which
        is why the 16 passes in sd-results-3687904 do not mean what they look
        like. Properly tiled, megablox BEATS `lax.ragged_dot` at tokamax's
        design shapes (0.24x at Mixtral 8x7b, i.e. 4.1x faster) and at our probe
        shape (0.79x) -- so the right repair was to configure the denominator,
        not to replace it.
        """
        if jax.default_backend() != "tpu":
            type(self).baseline_impl = "lax-ragged-dot"
            return jax.lax.ragged_dot(lhs, rhs, group_sizes).astype(jnp.float32)
        try:
            from jax.experimental.pallas.ops.tpu.megablox import gmm

            out = gmm(lhs, rhs, group_sizes, tiling=_megablox_tiling).astype(jnp.float32)
            type(self).baseline_impl = "pallas-megablox-gmm"
            return out
        except Exception:
            type(self).baseline_impl = "lax-ragged-dot-fallback"
            return jax.lax.ragged_dot(lhs, rhs, group_sizes).astype(jnp.float32)

    def baseline_candidates(self):
        """Tuned Pallas megablox vs XLA `ragged_dot`. MEASURED (job 3691513):
        megablox wins 4.1x at the 8x7b orientation but LOSES to XLA at some
        probe shapes -- so the denominator has to be chosen per shape, or a
        candidate gets credit for beating whichever one we guessed."""
        return {
            "production": self.baseline,
            "xla-ragged-dot": lambda lhs, rhs, gs: jax.lax.ragged_dot(
                lhs, rhs, gs, preferred_element_type=jnp.float32),
        }

    def tp_specs(self, case=None):
        """Shard the OUTPUT feature axis n -- the classic MoE up-projection
        split. rhs [g, k, n] is sharded on n, lhs is replicated, out [m, n] is
        sharded on n, and no collective is needed. group_sizes is replicated
        because every shard sees the same row grouping."""
        from jax.sharding import PartitionSpec as P

        return ((P(), P(None, None, "tp"), P()), P(None, "tp"))

    def grad_calibration_variants(self):
        """The group-loop formulation (deliberately NOT a forward variant --
        see honest_variants) is the honest source of BACKWARD spread: its
        per-expert accumulation order differentiates to a genuinely different
        reverse-mode reduction, exhibiting the ~3e-3 bf16-scale gradient
        drift that a floor-collapsed band rejected on v5p (job 3722139).
        Cost is g x FLOPs on ONE fixture at boot -- calibration only."""
        return [_honest_group_loop_bf16]

    def honest_variants(self):
        """Deliberately EMPTY, and measured rather than assumed (v5p-8, job
        3689440). A GMM is a single reduction: the MXU accumulates in fp32
        whatever you ask for, so the fp32-accumulator path is bit-identical to
        the fp32 reference (error exactly 0.0) and contributes nothing to the
        band, while the bf16 path costs exactly one output rounding -- measured
        equal to reference_bf16 to 4 s.f. (3.101e-3 vs 3.101e-3), which
        reference_bf16 already models and deliberately allows. There is no
        accumulation chain here to widen the band for, unlike attention, FLCE
        and rg_lru.

        `_honest_group_loop_bf16` is kept below for reference but NOT returned:
        it dots every row against every expert and masks, which is g times the
        FLOPs -- harmless at the probe set's g=4, but 64x at the production
        `32k-e64` cases, i.e. ~246 TFLOP burned per calibration to widen the
        band by nothing.
        """
        return []

    def adversarial_cases(self):
        # settable base case: a judge grading a non-default (e.g. probe) case
        # set must not silently force candidates to also trace at the tiny
        # CPU-battery shapes, which no prompt declares
        tiny = self.case_by_name(self.adversarial_case_name)

        def empty_group(key):
            lhs, rhs, sizes = self.make_inputs(key, tiny)
            m = lhs.shape[0]
            sizes = jnp.array([0, m // 2, 0, m - m // 2], jnp.int32)
            return (lhs, rhs, sizes)

        def max_skew(key):
            lhs, rhs, sizes = self.make_inputs(key, tiny)
            m = lhs.shape[0]
            sizes = jnp.array([m, 0, 0, 0], jnp.int32)  # single-expert pileup
            return (lhs, rhs, sizes)

        def expect_finite(ref, inputs):
            assert np.isfinite(np.asarray(ref, np.float64)).all()

        def expect_skew_exact(ref, inputs):
            lhs, rhs, sizes = inputs
            want = np.asarray(lhs, np.float32) @ np.asarray(rhs, np.float32)[0]
            np.testing.assert_allclose(np.asarray(ref), want, rtol=2e-4, atol=2e-4)

        return [
            AdversarialCase("empty-expert-groups", empty_group, expect_finite),
            AdversarialCase("max-skew-single-expert", max_skew, expect_skew_exact),
        ]


PROBLEM = MegabloxGmmProblem()
