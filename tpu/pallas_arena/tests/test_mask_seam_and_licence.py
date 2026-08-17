"""Two seams that keep the splash contract honest, both CPU-checkable.

MASK SEAM. Our reference implements the sliding window as a predicate
(`i - j < window`, causal); the production baseline binds JAX splash's
`LocalMask(window_size=(window-1, 0))`. Those are two independent encodings
of one convention, and nothing pinned them to each other -- the interval
mapping (inclusive-of-self vs not) was verified by hand once, in a session,
which is exactly the kind of fact that silently rots. These tests compare
the MASK ARRAYS themselves, so a convention drift fails loudly at mask
level instead of surfacing as a mysterious tolerance miss on TPU.
(tokamax's experimental splash carries a 1755-line mask suite for the same
machinery; this is the slice of it our contract actually leans on.)

LICENCE TEST. The blocked implementations (`_xla_grouped_attention`,
`_xla_masked_attention`) are candidates for ANSWER-KEY duty at shapes where
the closed form cannot fit (s=16384 needs 32 GB). What licenses that
substitution is agreement with the closed form everywhere both can run --
so that agreement is a standing assertion, not a one-off measurement. Both
paths are pure fp32, so the bound is fp-rounding-tight (1e-5), far inside
the calibrated grading band: an edit to the blocking logic that changes the
function fails here before it can redefine correctness at the big shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pallas_arena.judge.problems import get_problem
from pallas_arena.judge.problems.base import error_stats
from pallas_arena.judge.problems.splash_attention import (
    _xla_grouped_attention,
    _xla_masked_attention,
    causal_segment_attention,
)


def _our_window_mask(seq: int, window: int | None) -> np.ndarray:
    i = np.arange(seq)[:, None]
    j = np.arange(seq)[None, :]
    m = i >= j
    if window is not None:
        m &= (i - j) < window
    return m


# ------------------------------------------------------------------ mask seam
@pytest.mark.parametrize("seq,window", [(128, 1), (128, 7), (128, 64), (256, 128), (256, 255)])
def test_localmask_matches_our_window_predicate(seq, window):
    """LocalMask(window_size=(window-1, 0)) must equal `causal & (i-j < window)`
    element for element. left=window-1 because LocalMask counts EXCLUSIVE
    neighbours while our `window` counts positions including self."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam

    theirs = np.array(sam.LocalMask((seq, seq), window_size=(window - 1, 0), offset=0)[:, :])
    np.testing.assert_array_equal(theirs, _our_window_mask(seq, window))


@pytest.mark.parametrize("seq", [128, 256])
def test_causalmask_matches_our_causal_predicate(seq):
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam

    theirs = np.array(sam.CausalMask(shape=(seq, seq))[:, :])
    np.testing.assert_array_equal(theirs, _our_window_mask(seq, None))


def test_window_at_least_seq_equals_causal_at_mask_level(seq=128):
    """The reference-level limit test exists (test_case_features); this is the
    same fact at MASK level, where an off-by-one is visible as one wrong cell
    rather than a small numeric drift."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam

    local = np.array(sam.LocalMask((seq, seq), window_size=(seq - 1, 0), offset=0)[:, :])
    causal = np.array(sam.CausalMask(shape=(seq, seq))[:, :])
    np.testing.assert_array_equal(local, causal)


# ---------------------------------------------------------------- licence test
def _small_cases():
    p = get_problem("splash_attention")
    cases = [c for c in p.shape_cases() if c.smoke]
    # plus the structures the smoke set does not span, at licence-affordable size
    from pallas_arena.judge.problems.base import ShapeCase

    cases += [
        ShapeCase("lic-gqa", {"heads": 8, "kv_heads": 2, "seq": 192, "d": 32}),
        ShapeCase("lic-mqa", {"heads": 8, "kv_heads": 1, "seq": 192, "d": 32}),
        ShapeCase("lic-dv", {"heads": 4, "kv_heads": 4, "seq": 192, "d": 48, "d_v": 32}),
        ShapeCase("lic-h6kv2", {"heads": 6, "kv_heads": 2, "seq": 160, "d": 32}),
        ShapeCase("lic-window", {"heads": 4, "seq": 192, "d": 32},
                  features=(("window", 48),)),
        ShapeCase("lic-cap", {"heads": 4, "seq": 192, "d": 32},
                  features=(("soft_cap", 20.0),)),
    ]
    return p, cases


def test_blocked_implementations_agree_with_closed_form_everywhere_both_fit():
    import jax

    p, cases = _small_cases()
    for case in cases:
        pc = p.for_case(case)
        ins = p.make_inputs(jax.random.PRNGKey(0), case)
        ref = pc.reference(*ins)
        feats = case.feature_kwargs

        got = _xla_grouped_attention(*ins, **feats)
        assert error_stats(got, ref)["max"] < 1e-5, f"{case.name}: grouped diverged"

        # the square-MHA path only where it is expressible
        qh = ins[0].shape[0]
        kvh = ins[1].shape[0]
        dv = ins[2].shape[-1]
        if dv == ins[0].shape[-1]:
            got_m = _xla_masked_attention(*ins, **feats)
            assert error_stats(got_m, ref)["max"] < 1e-5, f"{case.name}: masked diverged"
        # and on MHA the two blocked paths are the same computation
        if qh == kvh and dv == ins[0].shape[-1]:
            same = error_stats(got, causal_segment_attention(*ins, **feats))["max"]
            assert same < 1e-5, f"{case.name}: grouped vs closed drifted ({same:.2e})"
