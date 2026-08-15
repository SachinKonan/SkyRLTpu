"""Can the REAL recurrentgemma scan be bound as the rg_lru denominator?

The arena's 1.1941 is "1.19x `lax.associative_scan`", because `rg_lru.baseline`
raises `BaselineUnavailable` on its production branch unconditionally -- a
deliberate Phase-2 stub, not a provisioning accident. This module asks the only
question that can upgrade that claim: **does a Pallas LRU scan exist in the
installed `recurrentgemma`, and can it be called with our shapes?**

It is deliberately dumb and read-only. It imports the package, walks its module
tree ONE level at the documented paths, reports what is actually there with
`inspect.signature`, and -- if a callable turns up -- tries it against our own
fp32 reference at a small shape. Anything else (a name that does not exist, a
kernel that needs TPU, a signature we cannot satisfy) is reported as such.

The distinction that matters for the report:

  * `module_missing`  -- the import path in the design doc does not exist at
    this version. Then the baseline can never be bound and the claim stays
    "versus associative_scan" permanently, whatever host we run on.
  * `tpu_only`        -- it exists and refuses on CPU. Then the claim is
    upgradable, but only on a TPU judge, and the work is a shape adapter.
  * `bound`           -- it exists and runs. Then we can re-grade against it.

Usage: JAX_PLATFORMS=cpu python -m pallas_arena.verify.rg_baseline_probe
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import traceback

# The paths worth trying, most-specific first. `lru_pallas_scan` is the one the
# arena's own docstring names; the rest are where a scan would plausibly live.
CANDIDATE_MODULES = (
    "recurrentgemma.jax.pallas",
    "recurrentgemma.jax.scan",
    "recurrentgemma.jax.layers",
    "recurrentgemma.jax",
    "recurrentgemma",
)
CANDIDATE_ATTRS = (
    "lru_pallas_scan",
    "linear_scan",
    "rnn_scan",
    "scan",
    "pallas_scan",
    "_lru_pallas_scan",
)


def _version() -> str:
    try:
        import importlib.metadata as md

        return md.version("recurrentgemma")
    except Exception as e:  # noqa: BLE001
        return f"<unknown: {type(e).__name__}>"


def probe_imports() -> dict:
    out: dict = {"version": _version(), "modules": {}, "callables": {}}
    try:
        importlib.import_module("recurrentgemma")
    except Exception as e:  # noqa: BLE001
        out["package_import"] = f"{type(e).__name__}: {e}"
        return out
    out["package_import"] = "ok"

    for modname in CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            out["modules"][modname] = f"MISSING ({type(e).__name__}: {str(e)[:120]})"
            continue
        names = sorted(n for n in dir(mod) if not n.startswith("__"))
        out["modules"][modname] = {
            "present": True,
            "n_names": len(names),
            "scan_like": [n for n in names if "scan" in n.lower() or "pallas" in n.lower()],
        }
        for attr in CANDIDATE_ATTRS:
            fn = getattr(mod, attr, None)
            if fn is None or not callable(fn):
                continue
            key = f"{modname}.{attr}"
            try:
                sig = str(inspect.signature(fn))
            except Exception as e:  # noqa: BLE001
                sig = f"<no signature: {type(e).__name__}>"
            out["callables"][key] = {"signature": sig, "doc": (inspect.getdoc(fn) or "")[:300]}
    return out


def try_call(path: str, b: int = 2, t: int = 64, d: int = 16) -> dict:
    """Call one discovered callable at a small shape, against our reference."""
    import jax
    import jax.numpy as jnp

    from pallas_arena.judge.problems import get_problem

    modname, attr = path.rsplit(".", 1)
    fn = getattr(importlib.import_module(modname), attr)
    problem = get_problem("rg_lru")
    case = problem.case_by_name(problem.adversarial_case_name)
    x, a, reset = problem.make_inputs(jax.random.PRNGKey(3), case)
    ref = problem.reference(x, a, reset)

    # recurrentgemma's scans are written for (batch, time, width) with a reset
    # mask and an h0; we try the orderings that a shape adapter would try.
    h0 = jnp.zeros((x.shape[0], x.shape[-1]), jnp.float32)
    attempts = [
        ("x,a,reset,h0", (x, a, reset, h0), {}),
        ("x,a,reset", (x, a, reset), {}),
        ("x,a,h0", (x, a, h0), {}),
        ("kw:x,a,reset,h0", (), {"x": x, "a": a, "reset": reset, "h0": h0}),
    ]
    tried = []
    for label, pos, kw in attempts:
        try:
            got = fn(*pos, **kw)
            got = got[0] if isinstance(got, tuple) else got
            jax.block_until_ready(got)
            err = float(jnp.max(jnp.abs(jnp.asarray(got, jnp.float32) - ref)))
            return {"status": "bound", "arg_form": label, "max_abs_err_vs_reference": err,
                    "shape": list(getattr(got, "shape", ())), "tried": tried}
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            tried.append({"arg_form": label, "error": msg})
    joined = " ".join(t["error"] for t in tried).lower()
    status = "tpu_only" if ("tpu" in joined or "mosaic" in joined or "unsupported platform" in joined) else "call_failed"
    return {"status": status, "tried": tried}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = probe_imports()
    report["calls"] = {}
    for path in list(report.get("callables", {})):
        try:
            report["calls"][path] = try_call(path)
        except Exception:  # noqa: BLE001
            report["calls"][path] = {"status": "probe_error", "traceback": traceback.format_exc()[-600:]}

    statuses = {v.get("status") for v in report["calls"].values()}
    if "bound" in statuses:
        report["verdict"] = "BINDABLE -- a real recurrentgemma scan ran; the denominator can be upgraded"
    elif "tpu_only" in statuses:
        report["verdict"] = "TPU-ONLY -- exists, refuses on CPU; upgradable only on a TPU judge"
    elif report["callables"]:
        report["verdict"] = "PRESENT-BUT-UNCALLABLE -- a scan exists, no argument form we tried worked"
    elif report.get("package_import") != "ok":
        report["verdict"] = "PACKAGE ABSENT -- recurrentgemma is not installed in this environment"
    else:
        report["verdict"] = "NO PALLAS SCAN -- the package imports but exposes no scan-like callable"

    print(json.dumps(report, indent=1, default=str)[:9000])
    print(f"\n=== VERDICT: {report['verdict']} ===")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
