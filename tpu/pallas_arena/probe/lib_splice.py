"""`from lib import ...` -- let a candidate keep parts of the seed by name.

The seed-improve prompt asks for the COMPLETE program back, so a model that
wants to retune one inner loop still has to retype everything around it. On
splash that is 353 lines to change 34, and the cost is not just tokens: the
qwen splash cell failed 3 times on syntax errors (one at line 310) and twice
by calling its own helpers with the wrong arity -- bookkeeping errors in code
it never meant to touch. It also drops the seed's comments on the way past,
which is where the (8,128) tiling invariant was written down.

So the seed's top-level definitions are offered as an importable namespace.
The candidate writes

    from lib import _forward, _fwd_body
    def _backward(...): ...        # the part it actually changed
    def kernel(...): ...

and this module splices the imported definitions back in before grading.

Rules, all enforced here rather than trusted:
  * only the SEED's own top-level names are importable -- never the reference,
    the baseline, or anything else the judge knows;
  * a name the candidate DEFINES wins over the same name imported, so it can
    import broadly and override selectively;
  * dependencies are pulled transitively, because "import _forward" plainly
    means "and whatever it needs to run";
  * the result is ordinary Python with no `lib` module in sight, so the judge,
    the pregate and the artifact all see one self-contained program.
"""

from __future__ import annotations

import ast


class SpliceError(Exception):
    """Raised when a candidate's imports cannot be satisfied."""


def _toplevel_units(tree: ast.Module) -> dict[str, ast.stmt]:
    """name -> the top-level statement that binds it."""
    out: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            # Tuple unpacking counts. `ROWS, LANES = 8, 128` binds two names at
            # top level, and registering only ast.Name targets silently omitted
            # both -- so the prompt never offered LANES and a candidate that
            # imported it (measured 2026-08-30, qwen splash idx=21) failed on a
            # name the seed plainly defines.
            for t in node.targets:
                for leaf in ast.walk(t):
                    if isinstance(leaf, ast.Name):
                        out[leaf.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def _names_used(node: ast.stmt) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _header(tree: ast.Module, src: str) -> str:
    """The seed's imports and __future__, which spliced definitions need."""
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return "\n".join(ast.get_source_segment(src, n) or "" for n in keep)


def available(seed_src: str) -> list[str]:
    """Names a candidate may import, in seed order."""
    return list(_toplevel_units(ast.parse(seed_src)))


def wanted(program: str) -> list[str]:
    """Names this program asks `lib` for (empty if it never imports lib)."""
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "lib":
            out.extend(a.name for a in node.names)
    return out


def splice(program: str, seed_src: str) -> str:
    """Return `program` with its `from lib import ...` resolved against the seed.

    A program that does not import lib is returned byte-identical, so this is
    safe to run over every candidate whether or not the feature was used.
    """
    if "lib" not in program:
        return program
    try:
        prog_tree = ast.parse(program)
    except SyntaxError:
        return program          # let the pregate report the syntax error itself

    asked: list[str] = []
    aliases: list[tuple[str, str]] = []      # (alias, real name)
    lib_imports: list[ast.stmt] = []
    for node in prog_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "lib":
            # `from lib import _fwd_body as fwd` is ordinary Python and models
            # write it unprompted -- it was 3 of the 4 lib failures in the first
            # cell that used the feature. Refusing it turned a working candidate
            # into a scored-zero one for no reason. Splice the real definition
            # and bind the alias to it.
            for a in node.names:
                asked.append(a.name)
                if a.asname:
                    aliases.append((a.asname, a.name))
            lib_imports.append(node)
    if not lib_imports:
        return program

    seed_tree = ast.parse(seed_src)
    units = _toplevel_units(seed_tree)
    unknown = [n for n in asked if n not in units]
    if unknown:
        raise SpliceError(
            f"lib has no {', '.join(sorted(unknown))}; available: {', '.join(units)}")

    # The candidate's OWN definitions win, so importing broadly and overriding
    # one function is the intended usage rather than a conflict.
    defined = set(_toplevel_units(prog_tree))

    # Transitive closure over the seed's internal references.
    need: list[str] = []
    seen: set[str] = set()
    stack = list(asked)
    while stack:
        name = stack.pop()
        if name in seen or name in defined or name not in units:
            continue
        seen.add(name)
        need.append(name)
        stack.extend(n for n in _names_used(units[name]) if n in units)

    pieces = [_header(seed_tree, seed_src)]
    # Emit in SEED order, not discovery order: the seed is a working module, so
    # its own order already satisfies every definition-before-use dependency.
    for name, node in units.items():
        if name in need:
            seg = ast.get_source_segment(seed_src, node)
            if seg:
                pieces.append(seg)

    for alias, real in aliases:
        pieces.append(f"{alias} = {real}")

    body_lines = program.splitlines()
    drop: set[int] = set()
    for node in lib_imports:
        for ln in range(node.lineno - 1, (node.end_lineno or node.lineno)):
            drop.add(ln)
    pieces.append("\n".join(l for i, l in enumerate(body_lines) if i not in drop))
    return "\n\n".join(p for p in pieces if p.strip()) + "\n"
