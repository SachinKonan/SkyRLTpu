"""Contract composer: model bodies in, runnable program out.

The load-bearing test is the round-trip: composing the scaffold's own naive
fills (rewritten as contract-style full defs) must produce a program that is
ast-valid, passes the same interpret-mode correctness the seam tests use, and
honors tunable overrides. The error paths must produce model-facing messages
(they become improvement-loop feedback verbatim).
"""

from __future__ import annotations

import os
import textwrap

os.environ.setdefault("PALLAS_INTERPRET", "1")
os.environ.setdefault("ARENA_RLIMIT_GB", "64")

import numpy as np
import pytest

from pallas_arena.probe import seam_scaffolds as ss
from pallas_arena.probe.contract_compose import (
    ContractError,
    compose_contract,
    contract_prompt_section,
    scan_scaffold,
)


def _naive_contract_block() -> str:
    """The RGLRU naive fills, expressed the way a model would under the
    contract: full defs with the exact scaffold signatures."""
    fwd = textwrap.indent(ss.RGLRU_NAIVE_FWD.strip("\n"), "")
    bwd = textwrap.indent(ss.RGLRU_NAIVE_BWD.strip("\n"), "")
    return (
        "def _fwd_body(x_ref, a_ref, reset_ref, h_ref, *, t):\n"
        + textwrap.indent(fwd, "") + "\n\n"
        "def _bwd_body(x_ref, a_ref, reset_ref, h_ref, g_ref, dx_ref, da_ref, *, t):\n"
        + textwrap.indent(bwd, "") + "\n"
    )


def test_scan_scaffold_finds_the_contract():
    c = scan_scaffold(ss.RGLRU_SCAFFOLD)
    assert set(c.required_defs) == {"_fwd_body", "_bwd_body"}
    assert c.required_defs["_fwd_body"] == ["x_ref", "a_ref", "reset_ref", "h_ref", "t"]
    assert "BLOCK_D" in c.tunables


def test_naive_roundtrip_is_interpret_correct():
    import jax
    import jax.numpy as jnp

    composed = compose_contract(ss.RGLRU_SCAFFOLD, _naive_contract_block())
    assert "pallas_call" in composed and "def kernel" in composed
    ns: dict = {}
    exec(compile(composed, "<composed>", "exec"), ns)
    kernel = ns["kernel"]

    b, t, d = 2, 64, 16
    key = jax.random.PRNGKey(0)
    kx, ka, kr = jax.random.split(key, 3)
    x = jax.random.normal(kx, (b, t, d), jnp.bfloat16)
    a = jax.nn.sigmoid(jax.random.normal(ka, (b, t, d))).astype(jnp.float32) * 0.98
    reset = jax.random.bernoulli(kr, 0.05, (b, t))

    got = np.asarray(kernel(x, a, reset))

    # reference recurrence
    xf = np.asarray(x, np.float32); af = np.asarray(a, np.float32)
    rf = np.asarray(reset)
    ref = np.zeros((b, t, d), np.float32)
    for bi in range(b):
        h = np.zeros((d,), np.float32)
        for ti in range(t):
            at = af[bi, ti] * (0.0 if rf[bi, ti] else 1.0)
            at = af[bi, ti] * (1.0 - float(rf[bi, ti]))
            h = at * h + np.sqrt(np.maximum(1.0 - at * at, 0.0)) * xf[bi, ti]
            ref[bi, ti] = h
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)


def test_tunable_override_lands_in_source():
    block = _naive_contract_block() + '\nTUNABLES = {"BLOCK_D": 256}\n'
    composed = compose_contract(ss.RGLRU_SCAFFOLD, block)
    assert "BLOCK_D = 256" in composed


def test_helpers_and_imports_are_injected():
    block = (
        "import math\n"
        "SQRT_EPS = 1e-20\n"
        "def _my_gate(a):\n    return a\n\n"
        + _naive_contract_block()
    )
    composed = compose_contract(ss.RGLRU_SCAFFOLD, block)
    assert "import math" in composed and "def _my_gate" in composed


def test_missing_def_is_a_precise_contract_error():
    with pytest.raises(ContractError, match="_bwd_body"):
        compose_contract(
            ss.RGLRU_SCAFFOLD,
            "def _fwd_body(x_ref, a_ref, reset_ref, h_ref, *, t):\n    h_ref[0] = a_ref[0]\n",
        )


def test_wrong_signature_is_a_precise_contract_error():
    bad = _naive_contract_block().replace(
        "def _fwd_body(x_ref, a_ref, reset_ref, h_ref, *, t):",
        "def _fwd_body(x, a, reset, h, *, t):", 1)
    with pytest.raises(ContractError, match="required signature"):
        compose_contract(ss.RGLRU_SCAFFOLD, bad)


def test_unknown_tunable_lists_the_allowed_set():
    block = _naive_contract_block() + '\nTUNABLES = {"BLOCK_Q": 64}\n'
    with pytest.raises(ContractError, match="BLOCK_D"):
        compose_contract(ss.RGLRU_SCAFFOLD, block)


def test_prompt_section_names_the_contract():
    sec = contract_prompt_section(ss.RGLRU_SCAFFOLD)
    assert "_fwd_body" in sec and "_bwd_body" in sec and "TUNABLES" in sec
