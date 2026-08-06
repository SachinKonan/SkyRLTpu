"""Phase 0 baseline-importability gate for the Pallas kernel arena.

Verifies that every reference/baseline in the arena slate imports at our
pinned JAX (uv.lock), on a CPU-only compute node, and records exactly where
each one imports from (login-CPU jax env vs needs-the-TPU-host-venv).

Stages (each runs in its own uv environment, see run_phase0.sbatch):
  main           dev+tunix env: splash, megablox (jax-bundled), vLLM-TPU RPA,
                 in-tree FLCE, XLA RMSNorm.
  recurrentgemma dev+jax env with recurrentgemma added (pip, falling back to
                 a shallow clone in scratch): RG-LRU Pallas scan.
  maxtext        best-effort pip maxtext for the vendored megablox gmm; the
                 jax-bundled megablox in stage `main` is the load-bearing
                 check (MaxText vendors that kernel).

Each stage appends one JSON object per baseline to --out (jsonl).
Exit code 0 iff every REQUIRED baseline in the stage imported.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TPU_INFERENCE_ROOT = REPO_ROOT / "third_party" / "tpu-inference"


def _record(out_path: Path, entry: dict) -> None:
    entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(out_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    status = "OK " if entry.get("ok") else "FAIL"
    print(f"[phase0] {status} {entry['baseline']}: {entry.get('import_path')} "
          f"({entry.get('version', '?')}) {entry.get('error', '')}")


def _git_rev(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _base_env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import jax
        import jaxlib

        info["jax"] = jax.__version__
        info["jaxlib"] = jaxlib.__version__
        info["jax_backend"] = jax.default_backend()
    except Exception as e:  # pragma: no cover
        info["jax_error"] = repr(e)
    return info


# --------------------------------------------------------------------------- main stage


def check_splash(out: Path) -> bool:
    entry = {"baseline": "splash_attention",
             "import_path": "jax.experimental.pallas.ops.tpu.splash_attention"}
    try:
        import jax

        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel as sak,
            splash_attention_mask as sam,
        )

        for sym in ("make_splash_mha", "SplashAttentionKernel", "BlockSizes"):
            assert hasattr(sak, sym), f"missing symbol {sym}"
        assert hasattr(sam, "CausalMask"), "missing CausalMask"
        # tiny CPU sanity: mask construction is pure python/numpy
        mask = sam.CausalMask(shape=(128, 128))
        assert mask.shape == (128, 128)
        entry.update(ok=True, version=f"jax {jax.__version__} (pip, uv.lock)",
                     where="login-CPU jax env",
                     symbols=["make_splash_mha", "CausalMask", "BlockSizes"])
    except Exception as e:
        entry.update(ok=False, error=f"{type(e).__name__}: {e}",
                     tb=traceback.format_exc(limit=4))
    _record(out, entry)
    return bool(entry.get("ok"))


def check_megablox_jax(out: Path) -> bool:
    entry = {"baseline": "megablox_gmm (jax-bundled; MaxText vendors this kernel)",
             "import_path": "jax.experimental.pallas.ops.tpu.megablox"}
    try:
        import jax
        from jax.experimental.pallas.ops.tpu import megablox as mbx

        assert hasattr(mbx, "gmm"), "missing gmm"
        entry.update(ok=True, version=f"jax {jax.__version__} (pip, uv.lock)",
                     where="login-CPU jax env", symbols=["gmm"])
    except Exception as e:
        entry.update(ok=False, error=f"{type(e).__name__}: {e}",
                     tb=traceback.format_exc(limit=4))
    _record(out, entry)
    return bool(entry.get("ok"))


def _import_tpu_inference_kernel():
    """Import the vLLM-TPU ragged_paged_attention v3 kernel from the
    third_party/tpu-inference checkout.

    Try the plain package import first; if the top-level tpu_inference
    __init__ needs TPU/vllm bits absent on CPU, fall back to registering
    parent packages by hand (their __init__ files are license-only) so only
    the kernel module executes.
    """
    sys.path.insert(0, str(TPU_INFERENCE_ROOT))
    try:
        mod = importlib.import_module(
            "tpu_inference.kernels.ragged_paged_attention.v3.kernel")
        return mod, "plain package import"
    except Exception as plain_err:
        # scrub partial imports
        for name in [n for n in sys.modules if n.startswith("tpu_inference")]:
            del sys.modules[name]
        pkg_root = TPU_INFERENCE_ROOT / "tpu_inference"
        parents = [
            ("tpu_inference", pkg_root),
            ("tpu_inference.kernels", pkg_root / "kernels"),
            ("tpu_inference.kernels.ragged_paged_attention",
             pkg_root / "kernels" / "ragged_paged_attention"),
            ("tpu_inference.kernels.ragged_paged_attention.v3",
             pkg_root / "kernels" / "ragged_paged_attention" / "v3"),
        ]
        for name, path in parents:
            m = types.ModuleType(name)
            m.__path__ = [str(path)]
            m.__package__ = name
            sys.modules[name] = m
        mod = importlib.import_module(
            "tpu_inference.kernels.ragged_paged_attention.v3.kernel")
        return mod, (f"parent-stub import (plain import failed: "
                     f"{type(plain_err).__name__}: {plain_err})")


def check_rpa(out: Path) -> bool:
    entry = {
        "baseline": "ragged_paged_attention (vLLM-TPU)",
        "import_path":
            "tpu_inference.kernels.ragged_paged_attention.v3.kernel "
            f"[{TPU_INFERENCE_ROOT}]",
    }
    try:
        mod, how = _import_tpu_inference_kernel()
        assert hasattr(mod, "ragged_paged_attention"), \
            "missing ragged_paged_attention"
        entry.update(ok=True, how=how,
                     version=f"third_party/tpu-inference @ {_git_rev(TPU_INFERENCE_ROOT)}",
                     where="login-CPU jax env (path import from third_party checkout)",
                     symbols=["ragged_paged_attention"])
    except Exception as e:
        entry.update(ok=False, error=f"{type(e).__name__}: {e}",
                     tb=traceback.format_exc(limit=4))
    _record(out, entry)
    return bool(entry.get("ok"))


def check_flce(out: Path) -> bool:
    entry = {"baseline": "FLCE custom_vjp (in-tree, commits 198f41fa/2e85086f)",
             "import_path": "skyrl.backends.tunix_backend.TunixBackend._flce_target_logprobs"}
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        from skyrl.backends.tunix_backend import TunixBackend

        flce = TunixBackend._flce_target_logprobs

        # tiny CPU numeric check vs the closed form
        rng = np.random.default_rng(0)
        n, h, v, tile = 13, 8, 97, 8
        hidden = jnp.asarray(rng.normal(size=(1, n, h)), dtype=jnp.float32)
        w = jnp.asarray(rng.normal(size=(h, v)), dtype=jnp.float32)
        tgt = jnp.asarray(rng.integers(0, v, size=(1, n)))
        got = flce(lambda x: x @ w, hidden, tgt, tile)
        logits = (hidden @ w).astype(jnp.float32)
        want = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), tgt[..., None], axis=-1)[..., 0]
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=2e-5, atol=2e-5)
        entry.update(ok=True, version=f"in-tree @ {_git_rev(REPO_ROOT)}",
                     where="login-CPU jax env (needs --extra tunix deps: "
                           "transformers/cloudpathlib/optax)",
                     numeric_check="fwd matches log_softmax closed form (13 tok, V=97)")
    except Exception as e:
        entry.update(ok=False, error=f"{type(e).__name__}: {e}",
                     tb=traceback.format_exc(limit=4))
    _record(out, entry)
    return bool(entry.get("ok"))


def check_xla_rmsnorm(out: Path) -> bool:
    entry = {"baseline": "XLA RMSNorm (jit-fused closed form)",
             "import_path": "jax (pure jnp expression, jax.jit)"}
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        def rmsnorm(x, g, eps=1e-6):
            var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1,
                           keepdims=True)
            return (x.astype(jnp.float32) * jax.lax.rsqrt(var + eps) * g
                    ).astype(x.dtype)

        x = jnp.ones((4, 128), jnp.float32) * 3.0
        g = jnp.ones((128,), jnp.float32)
        y = jax.jit(rmsnorm)(x, g)
        np.testing.assert_allclose(np.asarray(y), np.ones((4, 128)), rtol=1e-5)
        entry.update(ok=True, version=f"jax {jax.__version__} (pip, uv.lock)",
                     where="login-CPU jax env")
    except Exception as e:
        entry.update(ok=False, error=f"{type(e).__name__}: {e}",
                     tb=traceback.format_exc(limit=4))
    _record(out, entry)
    return bool(entry.get("ok"))


# ----------------------------------------------------------- recurrentgemma stage


def check_recurrentgemma(out: Path, scratch: Path) -> bool:
    entry = {"baseline": "RG-LRU Pallas scan (google-deepmind/recurrentgemma)"}
    how = None
    try:
        try:
            import recurrentgemma  # noqa: F401
            how = "pip package"
        except ImportError:
            clone = scratch / "recurrentgemma"
            if not (clone / "recurrentgemma").exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1",
                     "https://github.com/google-deepmind/recurrentgemma",
                     str(clone)],
                    check=True, capture_output=True, timeout=300)
            sys.path.insert(0, str(clone))
            import recurrentgemma  # noqa: F401
            how = f"shallow clone @ {_git_rev(clone)}"

        import pkgutil

        pallas_mods = []
        pkg = sys.modules["recurrentgemma"]
        for m in pkgutil.walk_packages(pkg.__path__, prefix="recurrentgemma."):
            if "pallas" in m.name or "scan" in m.name:
                pallas_mods.append(m.name)
        # Import the jax scan module(s) and find the pallas linear-scan entry.
        found = {}
        for name in pallas_mods:
            try:
                mod = importlib.import_module(name)
            except Exception as e:
                found[name] = f"import failed: {type(e).__name__}: {e}"
                continue
            syms = [s for s in dir(mod)
                    if any(k in s.lower() for k in ("scan", "lru", "pallas"))
                    and not s.startswith("_")]
            found[name] = syms
        version = getattr(sys.modules["recurrentgemma"], "__version__", "?")
        entry.update(ok=bool(found), how=how, version=str(version),
                     import_path="recurrentgemma (see modules field)",
                     modules=found,
                     where="importable in a CPU jax env with recurrentgemma "
                           "added (not in uv.lock; judge venv must add it)")
        if not found:
            entry.update(ok=False, error="no scan/pallas modules found")
    except Exception as e:
        entry.update(ok=False, how=how, error=f"{type(e).__name__}: {e}",
                     tb=traceback.format_exc(limit=4),
                     import_path="recurrentgemma")
    _record(out, entry)
    return bool(entry.get("ok"))


# ----------------------------------------------------------------- maxtext stage


def check_maxtext(out: Path) -> bool:
    entry = {"baseline": "megablox gmm (MaxText vendored copy) [best-effort]"}
    try:
        candidates = [
            "MaxText.kernels.megablox",
            "maxtext.kernels.megablox",
            "MaxText.kernels.megablox.gmm",
        ]
        last_err = None
        for name in candidates:
            try:
                mod = importlib.import_module(name)
                entry.update(ok=True, import_path=name,
                             version=getattr(mod, "__version__", "pip maxtext"),
                             where="CPU jax env with pip maxtext")
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err or ImportError("no maxtext module found")
    except Exception as e:
        entry.update(ok=False, error=f"{type(e).__name__}: {e}",
                     import_path="MaxText.kernels.megablox",
                     note="informational: MaxText is a manual TPU-host install "
                          "in this project; jax-bundled megablox (stage main) "
                          "is the arena baseline")
    _record(out, entry)
    return bool(entry.get("ok"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["main", "recurrentgemma", "maxtext"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scratch", type=Path,
                    default=Path(os.environ.get("PHASE0_SCRATCH", "/tmp")))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    _record(args.out, {"baseline": f"__env__ [{args.stage}]", "ok": True,
                       **_base_env_info()})

    if args.stage == "main":
        results = [
            check_splash(args.out),
            check_megablox_jax(args.out),
            check_rpa(args.out),
            check_flce(args.out),
            check_xla_rmsnorm(args.out),
        ]
        return 0 if all(results) else 1
    if args.stage == "recurrentgemma":
        return 0 if check_recurrentgemma(args.out, args.scratch) else 1
    if args.stage == "maxtext":
        check_maxtext(args.out)  # informational, never fails the gate
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
