"""New-env generation probe: Inkling-2-Mini BASE model (no adapter, no training), G groups x N samples of one prompt.

  inkling_probe.py --prompt probe_prompt.json --problem kakeya-128 --out .../inkling.jsonl --key 2 [--groups 2 --n 32]

Settings (user decision for the new envs: "rl28k generation settings for every model"): context 28,672; phase 1 (prompt +
thinking) 22,528; the answer gets the rest; temperature 1.0. The same two-phase completer as the Inkling runs
(InklingTwoPhaseTokenCompleter: thinking budget hit -> forced close + answer cue, masked). Prompt rendered with Inkling's
tml_v0 renderer at thinking effort 0.8, which puts its own "Thinking effort level" system message before the given
messages, exactly as the RL runs send it. prompt JSON: {"messages": [{"role": "system", ...}, {"role": "user", ...}]}.
One output row per completion: problem, model, group, index, prompt_tokens, completion_text, thinking_text,
answer_text, completion_tokens, thinking_tokens, answer_tokens, finish_reason (stop | length), thinking_hit_budget,
outcome (natural | injected | continued | wall), think_budget, group t_start / t_end / seconds, key, settings.
"""
import argparse, asyncio, json, os, subprocess, sys, time

MODEL = 'thinkingmachines/Inkling-2-Mini:peft:65536:free'
DISCOVER = os.environ.get('INKLING_PROBE_DISCOVER', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code/discover-0766e2413eb0')
CONTEXT, PHASE1, TEMPERATURE, EFFORT = 28672, 22528, 1.0, 0.8


def key(name):
    var = 'TINKER_API_KEY' if name == '1' else f'TINKER_API_KEY_{name}'
    return subprocess.run(['bash', '-ic', f'printf %s "${var}"'], capture_output=True, text=True).stdout


async def main(a):
    global CONTEXT, PHASE1
    CONTEXT, PHASE1 = a.context, a.phase1              # defaults = rl28k; smaller values only for tests
    os.environ['TTD_INKLING_EFFORT'] = str(EFFORT)
    for k in ('TTD_FIXED_THINK_TOKENS', 'TTD_FIXED_TOTAL_TOKENS', 'TTD_NATIVE_THINKING_BUDGET',
              'TTD_QWEN_SAMPLE_GROUP_CHUNK_SIZE'):
        os.environ.pop(k, None)                       # rl28k = legacy phase-1 budget, no fixed split
    sys.path.insert(0, DISCOVER)
    import tinker
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.tinker_utils.renderers import get_renderer
    from ttt_discover.tinker_utils.completers import InklingTwoPhaseTokenCompleter
    messages = json.load(open(a.prompt))['messages']
    tok = get_tokenizer(MODEL)
    renderer = get_renderer('tml_v0', tok)
    ob = renderer.build_generation_prompt(messages)
    stop = renderer.get_stop_sequences()
    sc = tinker.ServiceClient(api_key=key(a.key))
    client = await sc.create_sampling_client_async(base_model=MODEL)
    completer = InklingTwoPhaseTokenCompleter(sampling_client=client, tokenizer=tok, phase1_max_tokens=PHASE1,
                                              temperature=TEMPERATURE, context_window=CONTEXT, min_think_tokens=0)
    think_budget = PHASE1 - ob.length
    print(f'{a.problem}: prompt {ob.length} tokens, thinking budget {think_budget}, {a.groups} x {a.n}', flush=True)

    async def group(g):
        t0 = time.time()
        seqs = await completer.sample_group(ob, stop, a.n)
        return g, t0, time.time(), seqs

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_rows = 0
    with open(a.out, 'w') as fh:
        for g, t0, t1, seqs in await asyncio.gather(*(group(g) for g in range(a.groups))):
            for i, s in enumerate(seqs):
                toks = list(s.tokens)
                mask = list(getattr(s, 'maybe_mask', None) or [])
                if mask and 0.0 in mask:                   # forced close + cue were injected at the budget
                    p1 = mask.index(0.0)
                    outcome = 'wall' if all(m == 0.0 for m in mask[p1:]) else 'injected'
                    hit = True
                elif len(toks) >= think_budget:            # the budget ran out after the model opened its answer
                    p1, outcome, hit = think_budget, 'continued', True
                else:
                    p1, outcome, hit = len(toks), 'natural', False
                msg, stopped = renderer.parse_response(toks)
                answer_tokens = len(tok.encode(msg['content'], add_special_tokens=False)) if msg['content'] else 0
                if outcome == 'natural':                   # closed on its own: count the thinking text itself
                    p1 = len(tok.encode(msg['thinking'], add_special_tokens=False)) if msg['thinking'] else 0
                row = dict(problem=a.problem, model=MODEL, group=g, index=i, prompt_tokens=ob.length,
                           completion_text=tok.decode(toks), thinking_text=msg['thinking'], answer_text=msg['content'],
                           completion_tokens=len(toks), thinking_tokens=p1, answer_tokens=answer_tokens,
                           finish_reason='stop' if stopped else 'length', thinking_hit_budget=hit, outcome=outcome,
                           think_budget=think_budget, t_start=round(t0, 1), t_end=round(t1, 1),
                           seconds=round(t1 - t0, 1), key=a.key,
                           settings=dict(context=CONTEXT, phase1=PHASE1, temperature=TEMPERATURE, effort=EFFORT,
                                         renderer='tml_v0', completer='InklingTwoPhaseTokenCompleter', seed=None))
                fh.write(json.dumps(row) + '\n')
                n_rows += 1
    print(f'wrote {n_rows} rows -> {a.out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', required=True)
    p.add_argument('--problem', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--key', default='2', help='Tinker key name (base-model sampling works on any key)')
    p.add_argument('--groups', type=int, default=2)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--context', type=int, default=CONTEXT, help='rl28k default; change only for tests')
    p.add_argument('--phase1', type=int, default=PHASE1, help='rl28k default; change only for tests')
    asyncio.run(main(p.parse_args()))
