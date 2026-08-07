"""The repair experiment: how much of the export wall is ONE fact each?

For every seam candidate that died at `aot_export` on splash / GMM / RPA, apply
only the MECHANICAL repairs that correspond 1:1 to a fact a prompt could state,
re-compose against the same scaffold, and re-run the SAME CPU AOT pre-gate the
judge's first step is. A repair that flips a candidate from FAIL to PASS is
proof that the model's strategy was already good enough and the wall was a
single missing API fact -- and that the prompt fix is worth its tokens.

The four repairs, each named for the prompt line that makes it unnecessary:

  DN     `jax.lax.dot_general(dimension_numbers=...)` given FLAT
         `((1,), (0,), ...)` instead of `(((1,), (0,)), ((), ()))`.
  STORE  a SCALAR written into a Ref: `m_ref[...] = -jnp.inf`. A Pallas Ref
         store is not a broadcast; the RHS must already have the Ref's shape.
  WHEN   `@pl.when(c)` on a def, then CALLING that def. `pl.when` consumes the
         function and returns None.
  VMEM   `pltpu.VMEM(shape, dtype)` called inside a kernel body and indexed.
         It is a SPEC for `scratch_shapes=`, not an allocation.

Nothing here rewrites strategy: no block size, no loop structure, no algorithm.
If a repaired candidate still fails, the wall for that candidate is deeper than
a prompt line and is reported as such.

Usage: python -m pallas_arena.verify.repair_fills --results <jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys

from pallas_arena.probe import configs, pregate, seam

TASKS = ("splash_attention", "megablox_gmm", "ragged_paged_attention")


# ---------------------------------------------------------------- AST repairs
def _is_tuple_of_tuples(node) -> bool:
    return isinstance(node, ast.Tuple) and bool(node.elts) and all(isinstance(e, ast.Tuple) for e in node.elts)


def _scalarish(node) -> bool:
    """A value with no array shape: a number literal, +-inf, or -x of those."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _scalarish(node.operand)
    if isinstance(node, ast.Attribute) and node.attr in ("inf", "nan", "pi", "e"):
        return True
    return False


class Repairer(ast.NodeTransformer):
    def __init__(self, jnp_alias: str, pltpu_alias: str | None):
        self.jnp = jnp_alias
        self.pltpu = pltpu_alias
        self.applied: collections.Counter = collections.Counter()
        self._vmem_locals: set[str] = set()

    # -- DN ---------------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name == "dot_general":
            slot, dn = None, None
            for kw in node.keywords:
                if kw.arg == "dimension_numbers":
                    slot, dn = kw, kw.value
            if dn is None and len(node.args) >= 3:
                slot, dn = 2, node.args[2]
            if isinstance(dn, ast.Tuple) and not _is_tuple_of_tuples(dn):
                empty = ast.Tuple(elts=[ast.Tuple(elts=[], ctx=ast.Load()),
                                        ast.Tuple(elts=[], ctx=ast.Load())], ctx=ast.Load())
                if len(dn.elts) >= 2:
                    contract = ast.Tuple(elts=[dn.elts[0], dn.elts[1]], ctx=ast.Load())
                    new = ast.Tuple(elts=[contract, empty], ctx=ast.Load())
                    ast.fix_missing_locations(new)
                    if isinstance(slot, int):
                        node.args[slot] = new
                    else:
                        slot.value = new
                    self.applied["DN"] += 1
        # -- VMEM used as an allocation inside a body
        if name == "VMEM" and isinstance(fn, ast.Attribute):
            base = getattr(fn.value, "id", "")
            if self.pltpu is None or base == self.pltpu or base.endswith("pltpu"):
                # only rewritten when it is bound to a local (see visit_Assign)
                pass
        return node

    # -- STORE / VMEM ------------------------------------------------------
    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        # VMEM(...) bound to a local name inside a body -> a jnp value
        if (
            isinstance(tgt, ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "VMEM"
        ):
            self._vmem_locals.add(tgt.id)
            node.value = ast.Call(
                func=ast.Attribute(value=ast.Name(id=self.jnp, ctx=ast.Load()), attr="zeros", ctx=ast.Load()),
                args=list(node.value.args),
                keywords=[],
            )
            ast.fix_missing_locations(node)
            self.applied["VMEM"] += 1
            return node
        # `name[...] = v` where name is a VMEM local -> `name = v`
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id in self._vmem_locals:
            node.targets = [ast.Name(id=tgt.value.id, ctx=ast.Store())]
            ast.fix_missing_locations(node)
            self.applied["VMEM"] += 1
            return node
        # scalar into a Ref -> full_like
        if isinstance(tgt, ast.Subscript) and _scalarish(node.value):
            base = tgt.value
            base_name = getattr(base, "id", "") if isinstance(base, ast.Name) else ""
            if base_name.endswith("_ref") or base_name.endswith("_refs"):
                node.value = ast.Call(
                    func=ast.Attribute(value=ast.Name(id=self.jnp, ctx=ast.Load()),
                                       attr="full_like", ctx=ast.Load()),
                    args=[ast.Subscript(value=base, slice=tgt.slice, ctx=ast.Load()), node.value],
                    keywords=[],
                )
                ast.fix_missing_locations(node)
                self.applied["STORE"] += 1
        return node

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        tgt = node.target
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id in self._vmem_locals:
            new = ast.Assign(
                targets=[ast.Name(id=tgt.value.id, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=tgt.value.id, ctx=ast.Load()), op=node.op, right=node.value),
            )
            ast.fix_missing_locations(new)
            self.applied["VMEM"] += 1
            return new
        return node

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        # reads of a VMEM local: `acc[...]` -> `acc`
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in self._vmem_locals
            and isinstance(node.slice, ast.Constant)
            and node.slice.value is Ellipsis
        ):
            self.applied["VMEM"] += 1
            return ast.Name(id=node.value.id, ctx=ast.Load())
        return node

    # -- WHEN --------------------------------------------------------------
    def _strip_when_calls(self, body: list) -> list:
        whens = set()
        for st in body:
            if isinstance(st, ast.FunctionDef):
                for dec in st.decorator_list:
                    f = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(f, ast.Attribute) and f.attr == "when":
                        whens.add(st.name)
        out = []
        for st in body:
            if (
                isinstance(st, ast.Expr)
                and isinstance(st.value, ast.Call)
                and isinstance(st.value.func, ast.Name)
                and st.value.func.id in whens
                and not st.value.args
            ):
                self.applied["WHEN"] += 1
                continue
            out.append(st)
        return out

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        node.body = self._strip_when_calls(node.body)
        return node


def _aliases(tree: ast.Module) -> tuple[str, str | None]:
    jnp_alias, pltpu_alias = None, None
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "jax.numpy":
                    jnp_alias = a.asname or "jax.numpy"
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == "tpu" and "pallas" in (n.module or ""):
                    pltpu_alias = a.asname or "tpu"
    return jnp_alias or "jnp", pltpu_alias


def repair(fill_src: str) -> tuple[str, dict]:
    """Return (repaired source, {repair: count}). Raises on unparseable input."""
    tree = ast.parse(fill_src)
    jnp_alias, pltpu_alias = _aliases(tree)
    r = Repairer(jnp_alias, pltpu_alias)
    tree = r.visit(tree)
    ast.fix_missing_locations(tree)
    src = ast.unparse(tree)
    if r.applied and jnp_alias == "jnp" and "import jax.numpy as jnp" not in src:
        src = "import jax.numpy as jnp\n" + src
    return src, dict(r.applied)


# ------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--timeout-s", type=float, default=240.0)
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    rows = [json.loads(l) for l in open(args.results)]
    sigs = {t: pregate.probe_signatures(t, configs.TASK_CASES[t]) for t in tasks}

    out = {"per_candidate": [], "summary": {}}
    tally = collections.defaultdict(lambda: collections.Counter())

    for r in rows:
        if r["task"] not in tasks or r["variant"] != "seam":
            continue
        task = r["task"]
        fill = r.get("fill") or ""
        rec = {"task": task, "model": r["model"], "idx": r["idx"], "gate_before": r["gate"],
               "observation_before": (r.get("observation") or "").replace("\n", " | ")[:200]}
        tally[task]["n"] += 1
        if r["gate"] == "all":
            tally[task]["already_pass"] += 1
            rec["verdict"] = "already passed"
            out["per_candidate"].append(rec)
            continue
        try:
            fixed, applied = repair(fill)
        except SyntaxError as e:
            rec["verdict"] = f"unparseable fill: {e}"
            tally[task]["unparseable"] += 1
            out["per_candidate"].append(rec)
            continue
        rec["repairs"] = applied
        if not applied:
            rec["verdict"] = "no mechanical repair applies"
            tally[task]["no_repair"] += 1
            out["per_candidate"].append(rec)
            continue
        code = seam.compose(task, fixed)
        res = pregate.pregate_one(task, code, sigs[task], timeout_s=args.timeout_s)
        rec["gate_after"] = res.get("gate")
        rec["passed_after"] = bool(res.get("passed"))
        rec["observation_after"] = (res.get("observation") or "").replace("\n", " | ")[:200]
        rec["verdict"] = "REPAIRED -> exports" if res.get("passed") else "still fails"
        tally[task]["repaired_attempted"] += 1
        tally[task]["repaired_pass" if res.get("passed") else "repaired_fail"] += 1
        for k, v in applied.items():
            tally[task][f"applied:{k}"] += v
        out["per_candidate"].append(rec)
        print(f"[{task}] {r['model']} i{r['idx']}  {r['gate']} + {applied} -> "
              f"{'EXPORTS' if res.get('passed') else res.get('gate')}  {rec['observation_after'][:120]}")

    print("\n=== SUMMARY ===")
    for task in tasks:
        t = tally[task]
        before = t["already_pass"]
        after = before + t["repaired_pass"]
        print(f"{task:26s} n={t['n']:2d}  export before={before}  after mechanical repair={after}  "
              f"({dict(t)})")
        out["summary"][task] = dict(t)

    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
