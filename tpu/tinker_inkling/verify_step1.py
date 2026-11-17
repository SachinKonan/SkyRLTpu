"""Step-1 verification sheet for a held grid (run after every seed reached step 1, before RELEASE).

  verify_step1.py GRID [--out SHEET.md]

Checks each seed run against the agreed settings (REVIEW.md) and summarizes what actually happened in step 1:
settings in effect (env.txt), the prompt as sent (re-rendered from the run's own start state and settings),
thinking lengths / forced closes / continuations (client log), validity and best C5 (metrics), grading (journal),
training timing, and Tinker spend. Any mismatch is flagged FAIL at the top.
"""
import csv, json, os, re, statistics as S, subprocess, sys, tempfile, time
from pathlib import Path

RUNS = Path(os.environ.get('GRID_RUNS', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'))
CODE = Path('/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code')
EXPECT = {  # agreed settings (REVIEW.md, 2026-11-16)
    'TTD_ENSEMBLE_MODELS': 'thinkingmachines/Inkling-2-Small:tml_v0:inkling', 'TTD_INKLING_EFFORT': '0.8',
    'TTD_FIXED_THINK_TOKENS': '24000', 'TTD_FIXED_TOTAL_TOKENS': '34000', 'CONTEXT_WINDOW': '65536',
    'LEARNING_RATE': '2e-5', 'LORA_RANK': '32', 'GROUPS_PER_BATCH': '16', 'GROUP_SIZE': '32',
    'TTD_LOSS_FN': 'importance_sampling', 'TTD_ADV_ESTIMATOR': 'piecewise_valid_entropic_centered',
    'TTD_ADV_PIECEWISE_ALPHA': '1.0', 'TTD_REJECT_TRUNCATED': '1', 'TTD_FB_GROUPS_PER_REQUEST': '4',
    'TTD_OPTIM_EARLY': '1', 'TTD_GRADE_BUDGET_S': '1000', 'EVAL_TIMEOUT': '1100', 'TTD_ERDOS_PROMPT_CPUS': '1',
    'TTD_ERDOS_RECORD': '0.3808563', 'TTD_ERDOS_CVXPY_SOLVERS': 'CLARABEL, SCS, OSQP, HIGHS, SCIPY',
    'TTD_ERDOS_PROMPT': 'v2', 'TTD_SAFE_GRADE_MAX_WORKERS': '512', 'TTD_EVAL_BACKEND': 'farm',
    'KL_PENALTY_COEF': '0', 'TTD_FIXED_BUDGET_FIT': '1', 'TTD_M0_CONTEXT_WINDOW': '65536',
    'TTD_M0_TRAIN_MAX_SEQ': '65536', 'TTD_M0_PHASE1_MAX_TOKENS': '65536', 'TTD_GRADE_WAIT_TIMEOUT': '172800', 'TTD_ERDOS_ANSWER_RULES': '0',
}
PER_SEED = ('TTD_M0_LORA_SEED', 'TTD_INIT_STATE_SEED')   # must equal the seed number i of run <grid>-s<i>


def env_of(run):
    out = {}
    for line in (run / 'env.txt').read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k] = v
    return out


def thinking_stats(log_text):
    natural = [int(x) for x in re.findall(r'two_phase outcome=natural phase1=(\d+)', log_text)]
    injected = len(re.findall(r'two_phase outcome=injected', log_text))
    continued = len(re.findall(r'two_phase outcome=continued', log_text))
    return natural, injected, continued


def grid_runs(grid):
    """[(run dir, seed, frozen)] the grid starts, from the settings it was last started with
    (RUNS/grid-<G>/service.args: GRID_TARGETS, GRID_FROZEN); default 9 seeds, no controls."""
    args = {}
    f = RUNS / f'grid-{grid}' / 'service.args'
    if f.exists():
        args = dict(l.split('=', 1) for l in f.read_text().splitlines() if '=' in l)
    pairs = lambda v: [int(kv.split(':')[0]) for kv in v.split(',') if kv]
    seeds = pairs(args['GRID_TARGETS']) if args.get('GRID_TARGETS') else list(range(1, 10))
    frozen = pairs(args.get('GRID_FROZEN', ''))
    return ([(RUNS / f'{grid}-s{i}', i, False) for i in seeds] + [(RUNS / f'{grid}-f{i}', i, True) for i in frozen])


def main():
    grid = sys.argv[1]
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    plan = grid_runs(grid)
    seeds = [r for r, _, _ in plan]
    lines, fails = [], []
    lines.append(f'# Step-1 verification — grid {grid} — {time.strftime("%Y-%m-%d %H:%MZ", time.gmtime())}\n')
    lines.append('| seed | step | settings | valid | best C5 (x1e-6 > 0.38085) | natural-finish tokens p50/p90 | '
                 'forced / continued | graded | grade timeouts | groups failed | truncation drops | restarts | train s | '
                 'grade seed |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    grade_seeds, all_natural, code_seen = set(), [], set()
    for run, seed_i, frozen in plan:
        name = run.name
        lg = run / 'tinker_log' / name
        step = json.loads((lg / 'global_step.json').read_text())['global_step'] if (lg / 'global_step.json').exists() else 0
        env = env_of(run) if (run / 'env.txt').exists() else {}
        want = dict(EXPECT, **{k: str(seed_i) for k in PER_SEED})
        if frozen:   # control: same scaffold, no policy update
            want.update(LEARNING_RATE='0', TTD_FROZEN_POLICY='1')
        bad = [f'{k}={env.get(k)!r} (want {v!r})' for k, v in want.items() if env.get(k) != v]
        if not frozen and env.get('TTD_FROZEN_POLICY') not in (None, '0'):
            bad.append(f'TTD_FROZEN_POLICY={env.get("TTD_FROZEN_POLICY")!r} on a training run')
        ckpt = lg / 'member_inkling' / 'checkpoints.jsonl'
        has_rows = ckpt.exists() and ckpt.read_text().strip() != ''
        if frozen and (has_rows or not (lg / 'member_inkling' / 'frozen_policy_export.json').exists()):
            bad.append(f'frozen control: checkpoint rows {has_rows} (want none), frozen export '
                       f'{(lg / "member_inkling" / "frozen_policy_export.json").exists()} (want present)')
        if not frozen and step >= 1 and not has_rows:
            bad.append('training run at step >= 1 has no checkpoint rows')
        if bad:
            fails.append(f'{name}: ' + '; '.join(bad))
        if step < 1:
            fails.append(f'{name}: not at step 1 (step {step})')
        m = json.loads(open(lg / 'metrics.jsonl').readline()) if (lg / 'metrics.jsonl').exists() else {}
        valid = m.get('inkling/env/all/correctness')
        best = m.get('pool/best_value')
        log_text = (RUNS / f'{name}.log').read_text(errors='ignore') if (RUNS / f'{name}.log').exists() else ''
        natural, inj, cont = thinking_stats(log_text)
        timeouts = log_text.count('Timeout grading')
        exits = re.findall(r'client exited rc=(\S+)', log_text)
        # every client start prints one 'client pid' line; a kill/restart may end the wrapper before 'client exited'
        restarts = max(0, log_text.count('[launch_run] client pid') - 1)
        failed = m.get('inkling/rollout_groups_failed', 0) or 0
        dropped = m.get('inkling/reject_truncated/dropped_trajs', 0) or 0
        if timeouts:
            fails.append(f'{name}: {timeouts} grading timeout(s)')
        if failed:
            fails.append(f'{name}: {failed:.0f} rollout group(s) failed')
        if restarts:
            fails.append(f'{name}: {restarts} restart(s) before step 1 (exit codes seen: {exits})')
        # the code the run actually used (recorded by launch_run.sh) must be the deployed snapshot with all fixes
        code = {k: env.get(k, '') for k in ('CODE_SKYRL', 'CODE_DISCOVER')}
        if not all(code.values()):
            fails.append(f'{name}: env.txt has no CODE_SKYRL/CODE_DISCOVER (launched with old launch_run.sh)')
        else:
            dep = (CODE.parent / 'ops' / 'DEPLOYED').read_text().split()
            want_sk, want_di = dep[-2].split('=')[1][:12], dep[-1].split('=')[1][:12]
            if not code['CODE_SKYRL'].endswith(want_sk) or not code['CODE_DISCOVER'].endswith(want_di):
                fails.append(f'{name}: ran {code} but deployed is skyrl {want_sk} discover {want_di}')
            ens = Path(code['CODE_DISCOVER']) / 'ttt_discover' / 'rl' / 'ensemble.py'
            if 'TTD_GRADE_WAIT_TIMEOUT' not in ens.read_text():
                fails.append(f'{name}: client code {code["CODE_DISCOVER"]} lacks the grade-wait timeout fix')
            code_seen.add(tuple(code.values()))
        all_natural += natural
        jn = run / 'grading-journal.jsonl'
        graded = sum(1 for l in open(jn) if '"event": "result"' in l) if jn.exists() else 0
        gs = env.get('TTD_GRADE_SEED')
        grade_seeds.add(gs)
        q = lambda a, p: sorted(a)[min(len(a) - 1, int(p * len(a)))] if a else 0
        lines.append(f'| {name} | {step} | {"OK" if not bad else "FAIL"} | {valid if valid is None else f"{valid:.0%}"} | '
                     f'{"-" if best is None else f"{(-best - 0.38085) * 1e6:.1f}"} | {q(natural, .5)}/{q(natural, .9)} | '
                     f'{inj} / {cont} | {graded} | {timeouts} | {failed:.0f} | {dropped:.0f} | {restarts} | '
                     f'{m.get("inkling/time/train", float("nan")):.0f} | {gs} |')
        if valid is not None and valid < 0.5:
            fails.append(f'{name}: validity {valid:.0%} < 50%')
    if len(code_seen) > 1:
        fails.append(f'seeds ran different code: {sorted(code_seen)}')
    if len(grade_seeds) != len(plan):
        fails.append(f'grade seeds not distinct per run: {grade_seeds}')
    # prompt as sent: re-render seed 1's first start state with its settings and deployed code
    try:
        s1 = seeds[0]
        env = env_of(s1)
        disc = env.get('CODE_DISCOVER') or f'{CODE}/discover'   # render with the code the run itself used
        tree = json.loads((s1 / 'tinker_log' / s1.name / 'puct_sampler_step_000000.json').read_text())
        st = tree['initial_states'][0]
        script = (f"import json,os,sys; os.environ.update(json.loads(sys.argv[1])); sys.path[:0]=['{disc}/examples/erdos_min_overlap','{disc}']\n"
                  "from env import ErdosMinOverlapEnv\nfrom ttt_discover.tinker_utils.state import State\n"
                  "from ttt_discover.tinker_utils.renderers import get_renderer\nfrom ttt_discover.tinker_utils.misc_utils import get_tokenizer\n"
                  "d=json.loads(sys.argv[2]); st=State.from_dict(d); e=object.__new__(ErdosMinOverlapEnv); e.initial_state=st\n"
                  "r=get_renderer('tml_v0', get_tokenizer('thinkingmachines/Inkling-2-Small'))\n"
                  "mi=r.build_generation_prompt([{'role':'user','content':e.get_question()}]); print(r.tokenizer.decode(mi.to_ints()))")
        prompt = subprocess.run(['/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/python', '-W', 'ignore', '-c', script,
                                 json.dumps({k: v for k, v in env.items() if k.startswith('TTD_')}), json.dumps(st)],
                                capture_output=True, text=True, timeout=300).stdout
        checks = {'effort 0.8': re.search(r'effort level: 0\.8(?![0-9])', prompt) is not None, 'CPUs: 1': '**CPUs**: 1 available' in prompt,
                  'record 0.3808563': 'C₅ ≤ 0.3808563' in prompt, 'solver list': 'installed solvers: CLARABEL' in prompt,
                  'seed rule': 'fixed `seed` and `budget_s`' in prompt, 'lower is better': '(higher is better)' not in prompt,
                  'no v2 answer rules': 'only the final Python program' not in prompt}
        fails += [f'prompt: {k} missing' for k, ok in checks.items() if not ok]
        lines.append('\n## Prompt as sent (seed 1, start state 0)\n```\n' + prompt.strip()[:6000] + '\n```')
        lines.append('Prompt checks: ' + ', '.join(f'{k} {"OK" if ok else "FAIL"}' for k, ok in checks.items()))
    except Exception as e:
        fails.append(f'prompt render failed: {e!r}')
    q = lambda a, p: sorted(a)[min(len(a) - 1, int(p * len(a)))] if a else 0
    lines.insert(1, f'**{"ALL CHECKS PASSED" if not fails else f"{len(fails)} PROBLEM(S)"}**\n' +
                 ''.join(f'- FAIL {f}\n' for f in fails) +
                 f'\nCompletion length of rollouts that finished on their own (thinking + answer, all seeds): '
                 f'median {q(all_natural, .5)}, p90 {q(all_natural, .9)} tokens. Forced = hit the 24k thinking cap.\n')
    text = '\n'.join(lines)
    if out:
        Path(out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
