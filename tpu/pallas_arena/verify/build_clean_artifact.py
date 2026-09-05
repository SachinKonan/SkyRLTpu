"""Build the CLEAN 2x2 artifact: protocol, test suite, results, and every
single completion.

Reads only runs produced under the canonical configuration (RosettaStone
extraction, 32768 context, THINK_BUDGET=12288). Nothing from
runs/pallas_arena/void-preconical/ is admissible.

    python3 build_clean_artifact.py <out.html> [cell.jsonl:graded.json ...]

Each positional argument pairs a generations file with its graded verdicts.
The case list and adversarial vectors are introspected from the problem
modules so the page cannot drift from what the judge actually runs.
"""
from __future__ import annotations

import ast
import hashlib
import html
import json
import pathlib
import sys

REPO = pathlib.Path("/n/fs/vision-mix/sk7524/SkyRLTpu")
sys.path.insert(0, str(REPO / "tpu"))
sys.path.insert(0, str(REPO / "tpu/pallas_arena/probe"))

from gen_smoke import extract_completion  # noqa: E402
from pallas_arena.judge.problems import rg_lru, splash_attention  # noqa: E402
from pallas_arena.judge.problems.base import TOL_MULTIPLIER  # noqa: E402
from pallas_arena.probe.prompt_ref_first import (  # noqa: E402
    SEED_IMPROVE_TEMPLATE, build3seed, lib_section,
)
from pallas_arena.probe.smoke_config import CELLS  # noqa: E402

PROBLEMS = {"rg_lru": rg_lru.PROBLEM, "splash_attention": splash_attention.PROBLEM}
SEEDS = {
    "rg_lru": REPO / "tpu/pallas_arena/probe/seed_rglru_active.py",
    "splash_attention": REPO / "tpu/pallas_arena/probe/seed_splash_flash.py",
}
OBS = {
    "rg_lru": REPO / "runs/pallas_arena/seed-obs-rglru.txt",
    "splash_attention": REPO / "runs/pallas_arena/seed-obs-splash.txt",
}
# Verbatim copies of prompts as SERVED, keyed by nothing -- matched by sha.
PROMPT_ARCHIVE = REPO / "runs/pallas_arena/prompts"
SEED_REWARD = {"rg_lru": "1.000x", "splash_attention": "0.262x"}
# Read the SAME files run_seed_onestep.sh exported as $SEED_REWARD, so the
# rebuilt prompt is byte-identical to the served one instead of merely close.
SEED_REWARD_FILE = {
    "rg_lru": REPO / "runs/pallas_arena/seed-reward-rglru.txt",
    "splash_attention": REPO / "runs/pallas_arena/seed-reward-splash.txt",
}
SEED_NOTE = {
    "rg_lru": "seed is already at parity with the production kernel — a win here "
              "means genuinely beating recurrentgemma's scan",
    "splash_attention": "seed is ~3.8x slower than the production kernel — most of "
                        "the headroom is recovering ground, and >1.0 would beat it outright",
}
# The seed's OWN measured reward, and so the bar a candidate must clear to be
# an improvement. These are NOT both 1.0 and the report must not pretend they
# are: the rg_lru seed is at parity with the production kernel, while the
# splash seed is ~3x slower than it, which is most of that cell's headroom.
# The float is what the prompt's own "reward accrues only ABOVE this" line
# quotes (runs/pallas_arena/seed-reward-*.txt), so the page and the prompt the
# model actually read agree by construction.
#
# UPDATED 2026-08-28 to the MEASURED bar. The prompt quotes 0.262x for splash,
# but that came from an older grading whose scored set was 7 single-chip
# cases; the judge serving this run scores 10 (it adds tp4-gqa32x8-s4096,
# tp4-h32-s4096, tp4-mqa-h32kv1-s4096). Reward is a geomean over the scored
# cases, so the two are not comparable and "beat the seed" cannot be decided
# against the quoted figure. Re-grading the seed program itself through this
# judge (graded-seedbar-*.json) gives the like-for-like bar: splash 0.2316 on
# the same 10 cases, rg_lru 1.0000 on the same 5. What the model was TOLD
# stays in SEED_REWARD and is shown separately -- the page reports both.
SEED_BAR = {"rg_lru": 1.0000, "splash_attention": 0.2316}


def task_of(rows) -> str:
    """The problem these rows belong to, from the graded cell key 'task:variant'."""
    for r in rows:
        cell = r.get("cell") or ""
        if ":" in cell:
            return cell.split(":", 1)[0]
    return ""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


# ---------------------------------------------------------------- test suite
def case_rows(problem):
    """Every shape case the judge actually grades, introspected."""
    rows = []
    for c in problem.shape_cases():
        kind = []
        if getattr(c, "holdout", False):
            kind.append("holdout")
        if getattr(c, "probe", False):
            kind.append("probe")
        if c.name.startswith("tp"):
            kind.append(f"TP={c.name.split('-')[0][2:]}")
        rows.append({
            "name": c.name,
            "dims": dict(c.dims),
            "kind": ", ".join(kind) or "declared",
            "features": dict(getattr(c, "feature_kwargs", {}) or {}),
        })
    return rows


def adversarial_rows(problem):
    out = []
    for a in problem.adversarial_cases():
        out.append({"name": a.name, "doc": (a.check.__doc__ or "").strip().split("\n")[0]
                    if hasattr(a, "check") else ""})
    return out


# ------------------------------------------------------------------- results
def load_cell(gens_path: pathlib.Path, graded_path: pathlib.Path):
    gens = {}
    for line in gens_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gens[r.get("idx")] = r
    cells = json.loads(graded_path.read_text())
    rows = []
    for cell, d in cells.items():
        for gr in d.get("rows", []):
            g = gens.get(gr.get("idx"), {})
            rows.append({**gr, "text": g.get("text", ""), "family": g.get("family"),
                         "usage": g.get("usage"), "finish": g.get("finish_reason"),
                         "cell": cell})
    return rows


def program_of(row):
    try:
        return extract_completion(row.get("text") or "", ["kernel"], family=row.get("family"))
    except Exception:  # noqa: BLE001
        return None


def parses(p):
    if not p:
        return False
    try:
        ast.parse(p)
    except SyntaxError:
        return False
    return "def kernel" in p


# ---------------------------------------------------------------------- page
CSS = """
:root{--bg:#fbfaf7;--ink:#1a1a17;--dim:#5c5a52;--line:#ddd9cf;--card:#fff;
 --ok:#1f6f4a;--bad:#8f2f2f;--warn:#8a6a1f;--accent:#2f4f7f}
@media (prefers-color-scheme:dark){:root{--bg:#14140f;--ink:#eceae2;--dim:#a4a094;
 --line:#33322a;--card:#1c1c16;--ok:#63c295;--bad:#e08c8c;--warn:#d9b45f;--accent:#8fb4e8}}
:root[data-theme=dark]{--bg:#14140f;--ink:#eceae2;--dim:#a4a094;--line:#33322a;
 --card:#1c1c16;--ok:#63c295;--bad:#e08c8c;--warn:#d9b45f;--accent:#8fb4e8}
:root[data-theme=light]{--bg:#fbfaf7;--ink:#1a1a17;--dim:#5c5a52;--line:#ddd9cf;
 --card:#fff;--ok:#1f6f4a;--bad:#8f2f2f;--warn:#8a6a1f;--accent:#2f4f7f}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
 font:16px/1.6 ui-serif,Georgia,'Times New Roman',serif;}
main{max-width:60rem;margin:0 auto;padding:2.5rem 1.25rem 6rem}
h1{font-size:2rem;line-height:1.2;margin:0 0 .3rem;text-wrap:balance}
h2{font-size:1.35rem;margin:2.6rem 0 .6rem;padding-top:1.2rem;
 border-top:1px solid var(--line);text-wrap:balance}
h3{font-size:1.05rem;margin:1.4rem 0 .4rem}
p{margin:.6rem 0}
.sub{color:var(--dim)}
.mono,code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sm{font-size:.85em}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.9rem}
th,td{border-bottom:1px solid var(--line);padding:.4rem .5rem;text-align:left;
 vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:.8rem;letter-spacing:.03em;
 text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums;
 font-family:ui-monospace,monospace}
.wrap{overflow-x:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;
 padding:1rem 1.1rem;margin:1rem 0}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;
 padding:.8rem;overflow-x:auto;font-size:.78rem;line-height:1.45;max-height:34rem}
details{border:1px solid var(--line);border-radius:6px;margin:.5rem 0;
 background:var(--card)}
details>summary{cursor:pointer;padding:.55rem .8rem;font-size:.9rem;
 display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap}
details[open]>summary{border-bottom:1px solid var(--line)}
details .body{padding:.7rem .8rem}
.pass{color:var(--ok);font-weight:600}
.fail{color:var(--bad)}
.tag{font-size:.72rem;border:1px solid var(--line);border-radius:99px;
 padding:.05rem .5rem;color:var(--dim)}
.big{font-size:1.5rem;font-variant-numeric:tabular-nums;
 font-family:ui-monospace,monospace}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));gap:.8rem}
"""


def roofline_table(row):
    """Per-case timings plus whichever roofline resource BINDS this task.

    A task reports one side of the roofline, not both, and that is a
    property of the kernel rather than a gap in the data: splash_attention
    implements flops() (compute-bound, so MXU utilisation is the ceiling it
    is working against) and rg_lru implements bytes_moved() (memory-bound,
    so HBM speed-of-light is). Rendering both columns for both tasks fills
    half the table with em-dashes that read as "the judge failed to measure
    this", so the column that does not apply is dropped and named instead.
    """
    mx = row.get("mxu_fracs") or {}
    sol = row.get("speed_of_light_fracs") or {}
    lat = row.get("latencies") or {}
    if not (mx or sol or lat):
        return "<p class='sub sm'>no roofline recorded (candidate never reached timing)</p>"
    if mx and not sol:
        extra_h, kind = "<th>MXU cand / ref</th>", (
            "compute-bound: the ceiling is MXU utilisation, as % of chip peak bf16")
    elif sol and not mx:
        extra_h, kind = "<th>HBM cand</th>", (
            "memory-bound: the ceiling is HBM bandwidth, as % of chip peak")
    else:
        extra_h, kind = "<th>MXU cand / ref</th><th>HBM cand</th>", "both roofline sides recorded"
    names = sorted(set(mx) | set(sol) | set(lat))
    out = [f"<p class='sub sm'>{esc(kind)}</p>",
           "<div class='wrap'><table><tr><th>shape case</th>"
           "<th>cand ms</th><th>ref ms</th><th>ratio</th>" + extra_h + "</tr>"]
    for n in names:
        l = lat.get(n) or {}
        c = l.get("cand_median_s")
        r = l.get("ref_median_s")
        ratio = (r / c) if (c and r) else None
        m, s = mx.get(n), sol.get(n)
        cells = [f"<td class='mono sm'>{esc(n)}</td>",
                 f"<td class='num'>{f'{c*1e3:.3f}' if c else '&mdash;'}</td>",
                 f"<td class='num'>{f'{r*1e3:.3f}' if r else '&mdash;'}</td>",
                 f"<td class='num'>{f'{ratio:.3f}x' if ratio else '&mdash;'}</td>"]
        if "MXU" in extra_h:
            cells.append("<td class='num'>{}</td>".format(
                f"{100*m[0]:.1f}% / {100*m[1]:.1f}%"
                if isinstance(m, (list, tuple)) and len(m) >= 2 else "&mdash;"))
        if "HBM" in extra_h:
            cells.append("<td class='num'>{}</td>".format(
                f"{100*s:.0f}%" if isinstance(s, (int, float)) else "&mdash;"))
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table></div>")
    return "".join(out)


def completion_block(row, i):
    prog = program_of(row)
    ok = row.get("passed")
    rw = row.get("reward_with_bwd")
    verdict = (f"<span class='pass'>PASSED &nbsp;reward {rw:.4f}</span>"
               if ok and rw is not None else
               ("<span class='pass'>PASSED</span>" if ok else
                f"<span class='fail'>{esc(str(row.get('outcome'))[:90])}</span>"))
    u = row.get("usage") or {}
    meta = (f"<span class='tag'>idx {row.get('idx')}</span>"
            f"<span class='tag'>{esc(row.get('family'))}</span>"
            f"<span class='tag'>{u.get('completion_tokens','?')} completion tok</span>"
            f"<span class='tag'>finish {esc(row.get('finish'))}</span>"
            f"<span class='tag'>{'program parses' if parses(prog) else 'no usable program'}</span>")
    return f"""<details><summary>{verdict}{meta}</summary><div class="body">
{roofline_table(row)}
<h3>Extracted program <span class="sub sm">(what the judge graded)</span></h3>
<pre>{esc(prog or '-- nothing extractable --')}</pre>
<h3>Full completion <span class="sub sm">(verbatim, reasoning included)</span></h3>
<pre>{esc(row.get('text') or '')}</pre>
<h3>Judge observation <span class="sub sm">(the feedback an RL step would receive)</span></h3>
<pre>{esc(row.get('observation') or '(none)')}</pre>
</div></details>"""


def cell_section(title, rows):
    n = len(rows)
    task = task_of(rows)
    bar = SEED_BAR.get(task, 1.0)
    passed = [r for r in rows if r.get("passed")]
    rws = sorted(r["reward_with_bwd"] for r in passed if r.get("reward_with_bwd") is not None)
    beat = [r for r in rws if r > bar]
    beat_prod = [r for r in rws if r > 1.0]
    gates = {}
    for r in rows:
        if r.get("passed"):
            gates["PASSED"] = gates.get("PASSED", 0) + 1
        else:
            g = str(r.get("outcome") or "?")
            g = g[1:g.index("]")] if g.startswith("[") and "]" in g else g[:24]
            gates[g] = gates.get(g, 0) + 1
    head = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>{v}</td><td class='num'>{100*v/n:.0f}%</td></tr>"
        for k, v in sorted(gates.items(), key=lambda kv: -kv[1]))
    return f"""<h2>{esc(title)}</h2>
<div class="grid">
<div class="card"><div class="sub sm">candidates</div><div class="big">{n}</div></div>
<div class="card"><div class="sub sm">passed every gate</div><div class="big">{len(passed)}</div></div>
<div class="card"><div class="sub sm">best reward</div><div class="big">{(f'{rws[-1]:.4f}' if rws else '&mdash;')}</div></div>
<div class="card"><div class="sub sm">beat the seed (&gt;{bar:.3f})</div><div class="big">{len(beat)}</div></div>
</div>
<div class="wrap"><table><tr><th>outcome</th><th>n</th><th>share</th></tr>{head}</table></div>
<p class="sub">Rewards of survivors: <span class="mono">{esc([round(x, 4) for x in rws] or 'none')}</span>.
This seed scores <b>{bar:.3f}</b> against the production kernel, so <b>&gt;{bar:.3f}</b> means the model
improved on the program it was handed. Beating the production kernel itself needs &gt;1.0, which
{len(beat_prod)} candidate{'' if len(beat_prod) == 1 else 's'} did.</p>
<h3>Every completion</h3>
{''.join(completion_block(r, i) for i, r in enumerate(sorted(rows, key=lambda r: (not r.get('passed'), r.get('idx') or 0))))}"""


def honest_variant_rows() -> str:
    """The honest set, read off the problem objects instead of described.

    An earlier draft listed splash's variants (faithful-bf16, online-softmax)
    as though they were the suite's, which is wrong for rg_lru, whose honest
    freedom is scan REORDERING (associative / chunked). Introspecting means
    the page cannot drift from the band the judge actually computes.
    """
    out = []
    for task, prob in PROBLEMS.items():
        try:
            names = [getattr(f, "__name__", "?") for f in prob.honest_variants()]
        except Exception as exc:  # noqa: BLE001
            names = [f"(uninspectable: {type(exc).__name__})"]
        listed = ", ".join(f"<span class='mono'>{esc(n)}</span>" for n in names) or "none"
        out.append(f"<tr><td class='mono sm'>{esc(task)}</td><td class='sm'>{listed}</td>"
                   f"<td class='sm'>{'yes' if getattr(prob, 'baseline_calibrates', False) else 'no'}</td></tr>")
    return ("<div class='wrap'><table><tr><th>task</th><th>honest variants</th>"
            "<th>baseline widens band</th></tr>" + "".join(out) + "</table></div>")


def suite_section():
    parts = [f"""<h2>What every candidate is tested against</h2>
<p class="sub">The judge grades a candidate on real TPU silicon, never in CPU interpret mode &mdash;
interpret never runs Mosaic, and an entire corpus of kernels once passed it and then failed on chip.
Each (candidate, shape case) pair is graded in its <b>own fresh process on its own chip(s)</b>, so a
kernel that halts a core cannot poison another test.</p>
<h3>1. Correctness, against a true fp32 oracle</h3>
<p class="sub">Every case runs the candidate and the reference on the same inputs and compares
per-element error against a <b>calibrated tolerance</b>: {TOL_MULTIPLIER}&times; the worst error (max and q99 tails)
of any HONEST implementation of that task, floored at a small absolute epsilon. Calibrating against
honest implementations rather than a fixed epsilon is what lets a legitimately different numeric path
pass. The honest set is always the reference at bf16 working precision plus, per task:</p>
{honest_variant_rows()}
<p class="sub sm">These differ by task because the legitimate freedom differs: splash may reassociate a
softmax, rg_lru may reassociate a scan. A variant that cannot express a given shape, or that goes
non-finite on it, is dropped rather than allowed to widen the band.</p>
<p class="sub">The oracle itself is pinned to <span class="mono">Precision.HIGHEST</span>. Without that,
<span class="mono">jnp.einsum</span> multiplies f32 inputs through bf16 on TPU, and the arena was
scoring numerically exact kernels as WRONG &mdash; the seed measured ~1e-7 against a true f32 oracle
and ~3e-3 against the default-precision one. Four of six task oracles had this defect.</p>
<h3>2. Adversarial vectors</h3>
<p class="sub">Fixed hostile inputs every candidate must survive, checked the same way:</p>"""]
    for name, prob in PROBLEMS.items():
        advs = adversarial_rows(prob)
        if not advs:
            continue
        parts.append(f"<p class='sub sm'><b>{esc(name)}:</b> " +
                     ", ".join(f"<span class='mono'>{esc(a['name'])}</span>" for a in advs) + "</p>")
    parts.append("""<h3>3. Determinism</h3>
<p class="sub">The candidate is run 5 times on identical inputs and must be bitwise identical every
time; a kernel with a race passes correctness by luck otherwise.</p>
<h3>4. Backward, swept like the forward</h3>
<p class="sub">For tasks with a backward contract the judge differentiates a fixed scalar functional of
the kernel output with respect to its differentiable inputs, and compares against the same
differentiation of the reference &mdash; one gradient signature per shape case, features bound
identically to the forward. A wrong or missing backward FLOORS that case's factor; a correct backward
the judge could not time is excluded rather than counted, so absence can never beat slow-but-correct.</p>
<h3>5. Speed: 20 interleaved counterbalanced pairs, median of ratios</h3>
<p class="sub">Per case: 3 warm-up iterations, then <b>20 timed pairs</b>. Each pair runs BOTH the
reference and the candidate, alternating which goes first (R,C on even iterations, C,R on odd). The
score is the <b>median of the 20 per-pair ratios</b>, not a ratio of means.</p>
<p class="sub">The alternation is load-bearing and was measured: with a fixed reference-first order,
reference-vs-reference graded 1.019&ndash;1.053 and failed its own 1.00&plusmn;2% invariant, because the
first leg after fresh on-device input generation is systematically slower. Alternating spreads that
penalty over both roles so it cancels in the median, while interleaving still cancels drift.</p>
<h3>6. The noise floor, measured per chip</h3>
<p class="sub">At boot the judge times the reference <em>against itself</em> through the identical
protocol. That p95 spread is the chip's noise floor, and any score inside
<span class="mono">[1 - floor, 1 + floor]</span> collapses to exactly <b>1.0</b> &mdash; a tie is
reported as a tie. A floor above 0.5 means the chip could not tell a function from itself, and the case
is excluded rather than laundered into the reward.</p>
<h3>7. The reward</h3>
<p class="sub">Geometric mean over the scored cases' forward ratios, each gated by its own measured
floor, folded with one backward factor per case. Correct-everywhere: a candidate fault at ANY case
zeroes the candidate, because a kernel that is fast at one shape and wrong at another is not a kernel.
Judge faults (a measurement the judge could not make) exclude that case instead of punishing the
candidate.</p>""")
    # the concrete case list, introspected
    for name, prob in PROBLEMS.items():
        rows = case_rows(prob)
        body = "".join(
            "<tr><td class='mono sm'>{}</td><td class='mono sm'>{}</td><td class='sm'>{}</td></tr>".format(
                esc(r["name"]), esc(json.dumps(r["dims"])), esc(r["kind"]))
            for r in rows)
        parts.append(f"""<h3>Shape cases &mdash; {esc(name)} <span class="sub sm">({len(rows)} declared in the judge)</span></h3>
<div class="wrap"><table><tr><th>case</th><th>dims</th><th>role</th></tr>{body}</table></div>""")
    parts.append("""<p class="sub"><b>Holdouts</b> are graded but were never shown in the prompt, and
they are deliberately non-block-divisible (e.g. seq 1500, 2049) &mdash; that is what catches a kernel
tuned to the shapes it was given. <b>TP cases</b> run under <span class="mono">shard_map</span> across
4 chips with a per-shard baseline election.</p>""")
    return "".join(parts)


def _render_prompt(task: str, lib_imports: bool) -> str:
    """One prompt variant, built exactly the way gen_smoke.py builds it.

    The shape list MUST come from smoke_config.CELLS, which is what
    gen_smoke.main() passes to build3seed. Deriving it here instead from
    ``PROBLEM.shape_cases()`` looked equivalent and is not -- the judge
    declares cases the prompt never lists (holdouts especially) -- so that
    version rendered a prompt no model was ever shown.
    """
    cases, _kind = CELLS[(task, "rf3s")]
    seed = SEEDS[task].read_text()
    return SEED_IMPROVE_TEMPLATE.format(
        base=build3seed(task, cases, lib_imports=lib_imports),
        reward=SEED_REWARD_FILE[task].read_text().rstrip("\n"),
        program=seed,
        observation=OBS[task].read_text().strip() if OBS[task].exists() else "(none)",
        lib=lib_section(seed) if lib_imports else "")


def rebuild_prompt(task: str, served_sha: str | None = None) -> tuple[str, bool]:
    """(prompt, matched) -- the variant the cell was ACTUALLY served.

    There are now two variants of this prompt (with and without the lib-import
    offer), and a report that hardcodes one will sooner or later show a page of
    completions next to a prompt that produced none of them. So render both and
    pick the one whose sha256 matches the prompt_sha the sampler recorded; that
    turns "this is the prompt" from a claim into a check.

    Falls back to the plain variant when nothing matches, flagged as unmatched
    so the page says so rather than quietly implying a match.
    """
    # ARCHIVE FIRST. The prompt CODE moves -- the Output/Strategy contract was
    # rewritten for the lib-import variant -- so a prompt served last week can
    # stop being reproducible from today's source. When that happens the honest
    # artefact shows the archived text, not the nearest thing the current code
    # can build. Only a sha match is accepted, so a stale archive cannot be
    # passed off as the served prompt either.
    if served_sha:
        for f in sorted(PROMPT_ARCHIVE.glob("*.txt")) if PROMPT_ARCHIVE.is_dir() else []:
            t = f.read_text()
            if hashlib.sha256(t.encode()).hexdigest()[:12] == served_sha:
                return t, True
    for lib in (False, True):
        p = _render_prompt(task, lib)
        if served_sha and hashlib.sha256(p.encode()).hexdigest()[:12] == served_sha:
            return p, True
    return _render_prompt(task, False), False


def prompt_section(served_shas):
    parts = ["""<h2>The starting prompt</h2>
<p class="sub">One template, one prompt per task &mdash; both models saw byte-identical text, which is
the point of the experiment. It differs between tasks only in the shape table and the seed program
pasted into it. The seed is a <b>working, correct</b> kernel in both cases, so the model is asked to
make something faster, never to fix something broken; but the two seeds sit at very different
speeds, and that sets a different bar in each column.</p>
<div class="wrap"><table><tr><th>task</th><th>seed file</th><th>seed reward</th>
<th>what &gt;bar means</th></tr>"""]
    for task in PROBLEMS:
        bar = SEED_BAR.get(task, 1.0)
        parts.append(
            f"<tr><td class='mono sm'>{esc(task)}</td>"
            f"<td class='mono sm'>{esc(SEEDS[task].name)}</td>"
            f"<td class='num'>{bar:.3f}</td><td class='sm'>{esc(SEED_NOTE[task])}</td></tr>")
    parts.append("</table></div>")
    for task in PROBLEMS:
        want = served_shas.get(task)
        prompt, _matched = rebuild_prompt(task, want)
        got = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        if want is None:
            badge = f"<span class='tag'>sha {esc(got)} &mdash; no cell supplied</span>"
        elif got == want:
            badge = (f"<span class='tag'>sha {esc(got)}</span>"
                     f"<span class='pass'>matches what was served</span>")
        else:
            badge = (f"<span class='tag'>rebuilt {esc(got)}</span>"
                     f"<span class='fail'>DOES NOT match served {esc(want)}</span>")
        parts.append(
            f"""<details><summary><b>{esc(task)}</b> &mdash; full prompt, verbatim
<span class="tag">{len(prompt)} chars</span>{badge}</summary>
<div class="body"><pre>{esc(prompt)}</pre></div></details>""")
    return "".join(parts)


def main():
    out = pathlib.Path(sys.argv[1])
    pairs = sys.argv[2:]
    cells = []
    served_shas = {}
    for spec in pairs:
        gens_s, graded_s = spec.split(":", 1)
        gens_p = pathlib.Path(gens_s)
        rows = load_cell(gens_p, pathlib.Path(graded_s))
        title = pathlib.Path(gens_s).stem.replace("arm-gens-", "").replace("-", " · ")
        cells.append((title, rows))
        # The prompt_sha the sampler recorded per row IS the served prompt's
        # identity. Collect it so prompt_section can prove the text on the page
        # is the text the model read; disagreement inside one task is itself a
        # finding and must not be averaged away.
        for line in gens_p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            t, sha = r.get("task"), r.get("prompt_sha")
            if not (t and sha):
                continue
            if served_shas.setdefault(t, sha) != sha:
                print(f"WARNING: {t} was served >1 distinct prompt "
                      f"({served_shas[t]} and {sha})", file=sys.stderr)

    body = [f"<style>{CSS}</style>", "<main>",
            "<h1>Pallas kernel arena — the clean 2×2</h1>",
            """<p class="sub">Four cells, one protocol: qwen3.5-27B and gemma-4-31B, each asked once to
improve a working RG-LRU kernel and a working splash-attention kernel. Identical across all four:
32768 context, a 12288-token thinking budget with the answer guaranteed the remainder, extraction
through ttt_discover's RosettaStone with the model family recorded per row, and grading on real
silicon. Earlier runs under different budgets or extraction are void and are not shown.</p>""",
            prompt_section(served_shas), suite_section()]
    for title, rows in cells:
        body.append(cell_section(title, rows))
    if not cells:
        body.append("<h2>Results</h2><p class='sub'>No cells supplied yet.</p>")
    body.append("</main>")
    out.write_text("\n".join(body))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
