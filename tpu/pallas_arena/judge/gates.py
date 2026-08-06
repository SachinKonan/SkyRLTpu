"""Anti-cheat gates: AST screening and reference-module poisoning.

Two complementary layers (DESIGN.md adversarial addendum #4):
  * ``ast_gate`` — static screen of the candidate source: banned imports
    (wrapping the reference/baseline), banned dotted calls (e.g. jax.nn
    attention for attention tasks), and the per-problem pallas_call
    requirement. Fast, runs client-side in the AOT pre-gate too.
  * ``install_poison_stubs`` — runtime backstop, stronger than AST: in the
    grading child, AFTER the harness has taken direct references to
    everything it needs, every banned module prefix is replaced in
    sys.modules (and on its parent package attribute) by a poison module
    that raises on any attribute access. This catches obfuscated imports
    (importlib.import_module("jax." + x)) that no AST scan can see.
"""

from __future__ import annotations

import ast
import sys
import types


class ArenaBannedImport(ImportError):
    """Raised when candidate code touches a poisoned reference module."""


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruct a dotted name from an Attribute/Name chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def ast_gate(
    code: str,
    *,
    banned_import_prefixes: tuple[str, ...] = (),
    banned_call_names: tuple[str, ...] = (),
    require_pallas: bool = False,
) -> list[str]:
    """Return a list of violations (empty == pass)."""
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    saw_pallas_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches_prefix(alias.name, banned_import_prefixes):
                    violations.append(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:  # relative import inside a candidate: nonsense
                violations.append("relative import in candidate")
            elif _matches_prefix(mod, banned_import_prefixes):
                violations.append(f"banned import: from {mod}")
            else:
                # `from jax.experimental.pallas.ops.tpu import splash_attention`
                for alias in node.names:
                    full = f"{mod}.{alias.name}" if mod else alias.name
                    if _matches_prefix(full, banned_import_prefixes):
                        violations.append(f"banned import: from {mod} import {alias.name}")
        elif isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            if dotted:
                if dotted == "pl.pallas_call" or dotted.endswith(".pallas_call"):
                    saw_pallas_call = True
                last = dotted.split(".")[-1]
                for banned in banned_call_names:
                    # match full dotted name or its last component, so both
                    # `jax.nn.dot_product_attention` and an aliased
                    # `jnn.dot_product_attention` trip the gate
                    if dotted == banned or last == banned.split(".")[-1]:
                        violations.append(f"banned call: {dotted}")
                        break
        elif isinstance(node, ast.Name):
            if node.id == "pallas_call":
                saw_pallas_call = True

    if require_pallas and not saw_pallas_call:
        violations.append("no pallas_call found (this task requires a real Pallas kernel)")
    return sorted(set(violations))


class _PoisonModule(types.ModuleType):
    """Module stub that raises on any attribute access."""

    def __init__(self, name: str):
        super().__init__(name)
        object.__setattr__(self, "_poison_name", name)

    def __getattr__(self, item):
        if item in ("__path__", "__spec__", "__loader__", "__all__"):
            # keep the import system itself functional; submodule imports of
            # a poisoned package resolve via sys.modules, which is poisoned
            raise AttributeError(item)
        raise ArenaBannedImport(
            f"reference module '{object.__getattribute__(self, '_poison_name')}' "
            f"is off-limits to candidates (attribute {item!r})"
        )


def install_poison_stubs(prefixes: tuple[str, ...]) -> list[str]:
    """Poison every banned module prefix in the CURRENT process.

    Must be called only in the grading child, after the harness has bound
    direct references to the problem module, references, baselines, timers
    and RNG helpers it needs (existing references keep working; any future
    import/attribute lookup by candidate code hits the poison).

    Returns the list of module names poisoned (for logging).
    """
    poisoned: list[str] = []

    # 1) every already-imported module under a banned prefix
    for name in list(sys.modules):
        if _matches_prefix(name, prefixes):
            sys.modules[name] = _PoisonModule(name)
            poisoned.append(name)

    # 2) the prefix roots themselves, so a *future* first import gets poison
    for p in prefixes:
        if p not in sys.modules:
            sys.modules[p] = _PoisonModule(p)
            poisoned.append(p)

    # 3) sever parent-package attributes: `from jax.experimental.pallas.ops.tpu
    #    import splash_attention` resolves via getattr(parent, tail) when the
    #    parent is already imported — the parent attr must be poisoned too.
    for name in poisoned:
        if "." in name:
            parent_name, _, tail = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and not isinstance(parent, _PoisonModule):
                try:
                    setattr(parent, tail, sys.modules[name])
                except Exception:
                    pass
    return sorted(set(poisoned))
