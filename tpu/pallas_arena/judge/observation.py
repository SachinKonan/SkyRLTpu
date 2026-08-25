"""Curated, tail-truncation-proof observation strings for arena verdicts.

Why this exists
---------------
A judge verdict today is ``reward=0.0`` plus a gate label and a `violations`
list that is often a raw traceback or a 900-character Mosaic diagnostic. The
RL side (``ttt_discover.tinker_utils.state.State.to_prompt``) keeps only the
**last 500 characters** of an observation, so for the failure modes that
matter most — a scoped-VMEM OOM, a block/shape mismatch, a tolerance miss —
the one line carrying the number is scrolled off the top and the model is fed
stack noise instead.

So the judge curates the observation itself: gate name, the specific
violation, and the *measured numbers*, in a few hundred characters of pure
signal that survive tail truncation intact.

Contract
--------
``build_observation(result)`` is a **pure function of the verdict dict** — no
device, no candidate code, no hidden state. It never reveals anything the
candidate could not already derive from the public task spec: no hidden seed,
no reference output, no fixture data. Only the candidate's own numbers, the
public shapes, and the published tolerances/limits.

The string is at most ``max_chars`` (default 480 < 500) so a tail-truncating
consumer keeps all of it. When a single diagnostic is longer than the budget,
the *middle* is elided — never the head (which carries the gate) and never
the tail (which carries the measured numbers).
"""

from __future__ import annotations

import re

MAX_CHARS = 480

# ---------------------------------------------------------------- extractors

# "Scoped allocation with size 36.25M and limit 32.00M exceeded scoped vmem
#  limit by 4.25M."  (the exact phrasing measured in phase 2/4)
_VMEM = re.compile(
    r"Scoped allocation with size\s+([\d.]+[KMG]?)\s+and limit\s+([\d.]+[KMG]?)"
    r"\s+exceeded scoped vmem limit by\s+([\d.]+[KMG]?)",
    re.I,
)
# the buffer that blew it: "%_fwd.1 = bf16[16384,6144]{...}"
_VMEM_BUF = re.compile(r"=\s*([a-z0-9]+\[[\d,\s]*\])", re.I)

# error_stats: "shape mismatch (4, 18433, 128) vs (4, 18432, 128)"
_SHAPE_MISMATCH = re.compile(r"shape mismatch\s+(\([^)]*\))\s+vs\s+(\([^)]*\))", re.I)

# check_tolerance: "per-element max error 3.100e-02 exceeds calibrated
# tolerance 8.000e-03" / "per-element q99 error tail ... exceeds ..."
_TOL = re.compile(
    r"per-element (max|q99) error(?: tail)?\s+([\d.eE+-]+)\s+exceeds\s+calibrated tolerance\s+([\d.eE+-]+)",
    re.I,
)

# worker/child prefix a correctness violation with the fixture label:
# "h8-s4096#seed0: per-element max error ..." / "adv:label-in-tail: ..."
_FIXTURE = re.compile(r"^((?:adv:)?[A-Za-z0-9_.:#-]+?):\s")

# compile budget: "... exceeded the 90s compile budget (91.2s at unit 'fwd00')"
_BUDGET = re.compile(r"exceeded the\s+([\d.]+)s compile budget\s*\(([\d.]+)s(?:[^)]*at unit\s+'?\"?([^'\")]+))?", re.I)

# jax broadcast/shape complaints worth normalizing
_INCOMPAT = re.compile(r"(Incompatible shapes[^.;\n]*)", re.I)
_DIM_MISMATCH = re.compile(r"(dimensions? .{0,60}?must (?:be )?equal[^.;\n]*)", re.I)

# pallas block/grid complaints
_BLOCK = re.compile(r"((?:block|grid|BlockSpec)[^.;\n]{0,120})", re.I)

# last "Type: message" line of a traceback
_EXC_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Warning|NotImplemented[A-Za-z]*))\s*:\s*(.*)$")

# candidate frame: File "<candidate>", line 42, in kernel
_CAND_FRAME = re.compile(r'File "<candidate>", line (\d+)(?:, in (\S+))?')

_MOSAIC = re.compile(r"(Mosaic failed to compile TPU kernel:?\s*)(.*)", re.I)
_NOT_IMPL = re.compile(r"NotImplementedError:\s*(.*)")

_WS = re.compile(r"\s+")


def _flat(text: str, limit: int = 240) -> str:
    """One line, no long runs of whitespace, no absolute paths."""
    text = text.replace("\\n", "\n")
    text = re.sub(r"(/[\w.\-]+){2,}/", "", text)  # strip long file paths
    text = _WS.sub(" ", text).strip()
    return _elide(text, limit)


def _elide(text: str, limit: int) -> str:
    """Shorten by removing the MIDDLE. The head names the failure and the
    tail carries the numbers; both must survive."""
    if len(text) <= limit:
        return text
    if limit <= 5:
        return text[:limit]
    head = (limit - 5) * 2 // 3
    tail = limit - 5 - head
    return text[:head] + " ... " + text[len(text) - tail :]


def _exception_tail(tb: str) -> str:
    """From a traceback: the final `Type: message` plus, if present, the
    candidate's own line number (the only frame the author can act on)."""
    lines = [ln.rstrip() for ln in tb.replace("\\n", "\n").splitlines() if ln.strip()]
    exc = ""
    for ln in reversed(lines):
        m = _EXC_LINE.match(ln.strip())
        if m:
            exc = f"{m.group(1)}: {m.group(2)}"
            break
    if not exc and lines:
        exc = lines[-1]
    frame = ""
    for ln in reversed(lines):
        m = _CAND_FRAME.search(ln)
        if m:
            frame = f" [candidate line {m.group(1)}" + (f" in {m.group(2)}]" if m.group(2) else "]")
            break
    return _flat(exc + frame, 260)


def _size_str(s: str) -> str:
    return s.upper().replace("MI", "M")


def _vmem_fact(text: str) -> str | None:
    m = _VMEM.search(text)
    if not m:
        return None
    used, limit, over = (_size_str(g) for g in m.groups())
    fact = f"CompileTimeScopedVmemOom: {used} requested vs {limit} limit (over by {over})"
    b = _VMEM_BUF.search(text)
    if b:
        fact += f"; offending block {b.group(1).replace(' ', '')}"
    return fact


def _numeric_facts(text: str) -> list[str]:
    """Every measured number we know how to name, most specific first."""
    facts: list[str] = []
    v = _vmem_fact(text)
    if v:
        facts.append(v)
    m = _SHAPE_MISMATCH.search(text)
    if m:
        a = m.group(1).replace(" ", "")
        b = m.group(2).replace(" ", "")
        facts.append(f"output shape mismatch {a} vs required {b}")
    for m in _TOL.finditer(text):
        kind, got, tol = m.group(1), m.group(2), m.group(3)
        facts.append(f"{kind} err {_num(got)} vs tol {_num(tol)}")
    m = _BUDGET.search(text)
    if m:
        unit = f" at unit {m.group(3)}" if m.group(3) else ""
        facts.append(f"compile took {float(m.group(2)):.1f}s vs {float(m.group(1)):.0f}s budget{unit}")
    return facts


def _num(s: str) -> str:
    try:
        return f"{float(s):.3g}"
    except ValueError:
        return s


def _fixture_of(v: str) -> str:
    m = _FIXTURE.match(v.strip())
    return m.group(1) if m else ""


# ------------------------------------------------------------------- hints
# Deterministic, judge-side, contract-level statements. They add no
# information a careful reader of the task spec would not already have; they
# just put it where the failure is.
_HINTS = {
    "ast": "the entrypoint must be a self-contained kernel; the banned modules are never importable",
    "poison_stub": "banned module reached at runtime; write the kernel, do not call a library implementation",
    "exec": "module-level code must import and define the entrypoint without executing the kernel",
    "aot_export": "the SAME entrypoint must trace at EVERY declared shape (jax.export, no concrete data)",
    "pregate": "the SAME entrypoint must trace at EVERY declared shape (jax.export, no concrete data)",
    "compile_budget": "one kernel, one compile per declared shape, inside the budget",
    "determinism": "outputs must be bitwise identical across repeated runs of the same input",
    "gradient": "backward must match the fp32 reference; a raw pallas_call is not differentiable "
    "(wrap it in jax.custom_vjp)",
}
_VMEM_HINT = "scoped VMEM ceiling is 32MB per pallas_call block; budget rows*cols*bytes_per_element under it"


def build_observation(result: dict, max_chars: int = MAX_CHARS, hints: bool = True) -> str:
    """Compact structured observation for one verdict dict.

    Layout (newline separated, most specific fact last so a tail-truncating
    consumer keeps the numbers even in the pathological case):

        GATE <gate> | <problem> | <case/fixture>
        <violation, flattened>
        <measured numbers>
        need: <contract statement>
    """
    if not isinstance(result, dict):
        return "GATE worker | malformed verdict"

    gate = str(result.get("gate") or ("all" if result.get("passed") else "unknown"))
    problem = str(result.get("problem") or "?")
    violations = result.get("violations") or []
    if isinstance(violations, str):
        violations = [violations]
    raw = " | ".join(str(v) for v in violations)

    if result.get("passed"):
        return _elide(_pass_observation(result, problem), max_chars)

    fixture = ""
    for v in violations:
        fixture = _fixture_of(str(v))
        if fixture:
            break

    head = f"GATE {gate} | {problem}" + (f" | {fixture}" if fixture else "")

    facts = _numeric_facts(raw)

    # The human-readable violation, cleaned per gate.
    if gate in ("exec", "harness", "worker") or "Traceback" in raw:
        body = _exception_tail(raw)
    elif gate in ("aot_export", "pregate", "candidate_compile", "artifact_load", "fixtures"):
        body = _compile_body(raw)
    elif gate in ("correctness", "timed_output_correctness", "gradient"):
        body = _flat(re.sub(r"^\S+?:\s", "", raw), 200) if not facts else ""
    else:
        body = _flat(raw, 220)

    lines = [head]
    if body:
        lines.append(body)
    lines.extend(facts)

    if hints:
        hint = _VMEM_HINT if any(f.startswith("CompileTimeScopedVmemOom") for f in facts) else _HINTS.get(gate)
        if hint:
            lines.append("need: " + hint)

    out = "\n".join(lines)
    if len(out) <= max_chars:
        return out
    # Over budget: shrink the free-text body first, never the facts.
    over = len(out) - max_chars
    if body and len(body) > over + 20:
        lines[1] = _elide(body, len(body) - over)
        return "\n".join(lines)
    return _elide(out, max_chars)


def _compile_body(raw: str) -> str:
    """Trace/lower/compile diagnostics: keep the sentence that names the
    problem, drop the XLA HLO dump around it."""
    m = _NOT_IMPL.search(raw)
    if m:
        return _flat("NotImplementedError: " + m.group(1), 200)
    m = _MOSAIC.search(raw)
    if m:
        return _flat("Mosaic compile failed: " + m.group(2), 200)
    for rx in (_INCOMPAT, _DIM_MISMATCH):
        m = rx.search(raw)
        if m:
            return _flat(m.group(1), 200)
    if _VMEM.search(raw):
        return "pallas_call block does not fit in scoped VMEM"
    m = _BLOCK.search(raw)
    if m and len(raw) > 240:
        return _flat(m.group(1), 200)
    ex = _exception_tail(raw)
    return ex or _flat(raw, 200)


def _pass_observation(result: dict, problem: str) -> str:
    """A pass still needs an observation: the score and where the time went,
    so the next attempt has a direction rather than just a number."""
    reward = result.get("reward")
    parts = [f"GATE all | {problem} | PASS reward={reward:.4f}" if isinstance(reward, float) else f"GATE all | {problem} | PASS"]
    lat = result.get("latencies") or {}
    impls = result.get("baseline_impl_per_case") or {}
    for case, d in list(lat.items())[:3]:
        try:
            c = float(d["cand_median_s"]) * 1e3
            r = float(d["ref_median_s"]) * 1e3
            # NAME the thing being raced. "ref 0.22ms" is anonymous; knowing the
            # bar is XLA rather than a tuned Pallas kernel changes what is worth
            # trying, and it is the candidate's target rather than hidden state.
            who = f" [{impls[case]}]" if case in impls else ""
            parts.append(f"{case}: cand {c:.3f}ms vs ref{who} {r:.3f}ms ({r / c if c else 0:.3f}x)")
        except Exception:
            continue
    # ROOFLINE, per case. Which resource a kernel is actually against is what
    # decides the next optimization -- a candidate at 15% of both bandwidth
    # and MXU is latency/pipelining bound, and neither wider tiles nor fewer
    # bytes will help it. Every case is shown (the old [:2] cap hid exactly
    # the shapes that behave differently).
    sol = result.get("speed_of_light_fracs") or {}
    if sol:
        parts.append("HBM bandwidth used (% of chip peak): "
                     + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sol.items()))
    mxu = result.get("mxu_fracs") or {}
    if mxu:
        parts.append("MXU utilization (% of chip peak bf16, yours vs ref): "
                     + ", ".join(f"{k} {100 * v[0]:.0f}% vs {100 * v[1]:.0f}%"
                                 for k, v in mxu.items()))
    peak = result.get("peak_hbm_bytes")
    if peak:
        parts.append(f"peak HBM {peak / 1e9:.2f}GB")
    return "\n".join(parts)


def attach_observation(result: dict, **kw) -> dict:
    """Populate ``result['observation']`` in place and return the dict."""
    try:
        result["observation"] = build_observation(result, **kw)
    except Exception as e:  # an observation must never break a verdict
        result["observation"] = f"GATE {result.get('gate', '?')} | observation failed: {type(e).__name__}"
    return result
