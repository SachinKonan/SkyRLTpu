"""Fixed-output-contract composer: model bodies in, complete program out.

The contract inverts the scaffold relationship: instead of the model
re-emitting the whole program (scaffold + bodies) and us hoping the machinery
survived generation, the model outputs ONLY

  * the required function definitions (full `def`s, exact names/signatures),
  * optional module-level helpers (functions/constants it invents),
  * optional extra imports,
  * an optional TUNABLES = {...} dict overriding the scaffold's ALL-CAPS
    constants (block sizes etc. -- an explicit search dimension),

and THIS module assembles the fixed machinery around them. Measured
motivation (rounds 1-6 + repair): "no `kernel` function defined" (5-8/cell),
machinery syntax deaths, NameError body-fills, and dialect dodging all
disappear by construction; bodies are ~0.5-1.5k tokens vs 1.7-3.1k full
programs; improvement prompts embed bodies-only parents (rg_lru median 6k ->
~4.3k, and splash improvement flips from context-infeasible to fitting).

Diversity note: this keeps exactly the seam rf3s drew -- the model still
writes the entire algorithm; round-6 rf3s produced 32/32 distinct programs
under the same seam, so the ranking gradient survives (the fully-tailored
scaffold's collapse came from closing the ALGORITHM, not the machinery).

Deliberately importable WITHOUT jax: the RL client venv is CPU-only.
Composition is pure ast + text; executing the result is the judge's job.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


class ContractError(ValueError):
    """Raised with a model-facing message: what the contract wanted, what it got."""


@dataclass
class Contract:
    """What the scaffold requires of the model output."""

    required_defs: dict[str, list[str]]  # name -> exact positional arg names
    tunables: dict[str, str] = field(default_factory=dict)  # NAME -> default (source text)


def scan_scaffold(scaffold_src: str) -> Contract:
    """Derive the contract from the scaffold itself: required defs are those
    whose body is a bare `raise NotImplementedError(...)` (docstring allowed);
    tunables are module-level ALL-CAPS assignments."""
    tree = ast.parse(scaffold_src)
    required: dict[str, list[str]] = {}
    tunables: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body = [n for n in node.body if not (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str))]
            if len(body) == 1 and isinstance(body[0], ast.Raise):
                exc = body[0].exc
                if isinstance(exc, ast.Call) and getattr(exc.func, "id", "") == "NotImplementedError":
                    args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
                    required[node.name] = args
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id.isupper():
                tunables[t.id] = ast.get_source_segment(scaffold_src, node.value) or ""
    if not required:
        raise ValueError("scaffold has no NotImplementedError stubs to fill")
    return Contract(required_defs=required, tunables=tunables)


def _fn_source(src: str, node: ast.FunctionDef) -> str:
    seg = ast.get_source_segment(src, node)
    if seg is None:  # pragma: no cover -- 3.8+ always provides it for parsed src
        raise ContractError(f"could not extract source for def {node.name}")
    return seg


def compose_contract(scaffold_src: str, model_block: str) -> str:
    """Assemble scaffold + model contract output into one complete program.

    Raises ContractError with a precise, model-facing message on violation --
    the message becomes the judge-style feedback for the improvement loop.
    """
    contract = scan_scaffold(scaffold_src)
    try:
        tree = ast.parse(model_block)
    except SyntaxError as e:
        raise ContractError(f"your output is not valid Python: {e}") from e

    fns: dict[str, ast.FunctionDef] = {}
    helpers: list[str] = []
    imports: list[str] = []
    tunable_over: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in contract.required_defs:
                fns[node.name] = node
            else:
                helpers.append(_fn_source(model_block, node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(_fn_source(model_block, node))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "TUNABLES":
                if not isinstance(node.value, ast.Dict):
                    raise ContractError("TUNABLES must be a literal dict, e.g. TUNABLES = {\"BLOCK_D\": 256}")
                for k, v in zip(node.value.keys, node.value.values):
                    if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                        raise ContractError("TUNABLES keys must be string literals")
                    if k.value not in contract.tunables:
                        raise ContractError(
                            f"unknown tunable {k.value!r}; this scaffold's tunables are: "
                            f"{sorted(contract.tunables)}")
                    tunable_over[k.value] = ast.get_source_segment(model_block, v) or ""
            else:
                helpers.append(_fn_source(model_block, node))
        # anything else (class defs, exprs) -> helpers verbatim
        elif isinstance(node, (ast.ClassDef,)):
            helpers.append(_fn_source(model_block, node))

    missing = [n for n in contract.required_defs if n not in fns]
    if missing:
        raise ContractError(
            f"missing required function definition(s): {missing}. Output ONE python "
            f"block that defines exactly: "
            + "; ".join(f"def {n}({', '.join(a)})" for n, a in contract.required_defs.items()))
    for name, want_args in contract.required_defs.items():
        node = fns[name]
        got = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
        if got != want_args:
            raise ContractError(
                f"def {name} has arguments {got}; the required signature is "
                f"({', '.join(want_args)}) -- keep it EXACT, the machinery calls it by this shape")

    # --- splice ------------------------------------------------------------
    lines = scaffold_src.splitlines()
    s_tree = ast.parse(scaffold_src)

    # 1. replace each stub def block with the model's def (bottom-up so line
    #    numbers stay valid)
    stubs = [n for n in s_tree.body
             if isinstance(n, ast.FunctionDef) and n.name in contract.required_defs]
    for node in sorted(stubs, key=lambda n: n.lineno, reverse=True):
        repl = _fn_source(model_block, fns[node.name]).splitlines()
        lines[node.lineno - 1 : node.end_lineno] = repl
    out = "\n".join(lines)

    # 2. tunable overrides: rewrite the constant assignment lines
    for name, val in tunable_over.items():
        out, n = re.subn(rf"(?m)^{name}\s*=.*$", f"{name} = {val}", out, count=1)
        if n != 1:
            raise ContractError(f"internal: tunable {name} not found in scaffold")

    # 3. extra imports + helpers go right after the scaffold's import block
    inject = [s for s in imports + helpers if s.strip()]
    if inject:
        s_lines = out.splitlines()
        last_import = 0
        for i, l in enumerate(s_lines[:80]):
            if re.match(r"^(import |from )", l):
                last_import = i
        s_lines[last_import + 1 : last_import + 1] = [""] + inject + [""]
        out = "\n".join(s_lines)

    out += "" if out.endswith("\n") else "\n"
    try:
        ast.parse(out)
    except SyntaxError as e:  # a helper that broke module structure
        raise ContractError(f"composed program is not valid Python ({e}); "
                            "check your helper definitions are complete top-level statements") from e
    return out


def contract_prompt_section(scaffold_src: str) -> str:
    """The Output section for a contract prompt, derived from the scaffold."""
    c = scan_scaffold(scaffold_src)
    defs = "\n".join(
        f"  def {n}({', '.join(a)}):" for n, a in c.required_defs.items())
    tun = ", ".join(f"{k} (default {v})" for k, v in sorted(c.tunables.items()))
    return f"""## Output (FIXED CONTRACT -- do not repeat the scaffold)

The scaffold above is FIXED machinery; it will be assembled around your code
automatically. Output ONE fenced ```python block containing ONLY:

1. The required function definitions, with these EXACT names and signatures:
{defs}
2. (optional) module-level helper functions/constants your bodies call.
3. (optional) extra imports beyond the scaffold's (jax, jax.numpy as jnp,
   pallas as pl, pltpu, functools are already imported).
4. (optional) TUNABLES = {{...}} overriding the scaffold's constants
   [{tun}] -- tile-size choices are part of your search space.

Do NOT redefine `kernel`, the pallas_call machinery, or anything else from
the scaffold: only the listed functions are read from your output, and a
missing or signature-changed definition scores 0 with the reason fed back."""
