"""New-env generation probe: Inkling-2-Mini BASE model (no adapter, no training), G groups x N samples of one prompt.

  inkling_probe.py --prompt probe_prompt.json --problem kakeya-128 --out .../inkling.jsonl --key 2 [--groups 2 --n 32]

Settings = Inkling-2-Mini's OWN production generation recipe, exactly as the Inkling arms and the MOMO shims run it
(mc-s5 / mn9-s5 env.txt; user correction 2026-11-28: the rl28k split is for the on-prem models only):
  renderer tml_v0 at thinking effort 0.8 (it puts its own "Thinking effort level" system message before the messages);
  InklingTwoPhaseTokenCompleter, temperature 1.0; FIXED budget: 22,000 thinking + 10,000 answer (32,000 generated),
  context 65,536 (phase1 65,536); FIT=1 (a prompt too long for 32k shrinks thinking, never the answer);
  thinking budget hit -> forced close + answer cue (masked), the answer gets the rest of the 32,000.
prompt JSON: {"messages": [{"role": "system", ...}, {"role": "user", ...}]}.
Output: line 1 = {"header": true, "settings": {...}} (also written to meta.json next to the JSONL), then one row per
completion: problem, model, group, index, prompt_tokens, completion_text, thinking_text, answer_text,
completion_tokens, thinking_tokens, answer_tokens, finish_reason (stop | length), thinking_hit_budget,
outcome (natural | injected | continued | wall), think_budget, answer_budget, group t_start / t_end / seconds, key.
"""
import argparse, asyncio, json, os, subprocess, sys, time

MODEL = 'thinkingmachines/Inkling-2-Mini:peft:65536:free'
DISCOVER = os.environ.get('INKLING_PROBE_DISCOVER', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code/discover-0766e2413eb0')
# Inkling-2-Mini production recipe (mc-s5 / mn9-s5 env.txt)
CONTEXT, PHASE1, TEMPERATURE, EFFORT = 65536, 65536, 1.0, 0.8
FIXED_THINK, FIXED_TOTAL, FIXED_FIT = 22000, 32000, 1


def key(name):
    var = 'TINKER_API_KEY' if name == '1' else f'TINKER_API_KEY_{name}'
    return subprocess.run(['bash', '-ic', f'printf %s "${var}"'], capture_output=True, text=True).stdout


async def main(a):
    think, total = a.think, a.total                    # production defaults; smaller values only for tests
    os.environ.update(TTD_INKLING_EFFORT=str(EFFORT), TTD_FIXED_THINK_TOKENS=str(think),
                      TTD_FIXED_TOTAL_TOKENS=str(total), TTD_FIXED_BUDGET_FIT=str(FIXED_FIT), TTD_MIN_THINK_TOKENS='0',
                      TTD_NATIVE_THINKING_BUDGET='0', TTD_QWEN_TWO_PHASE='0')
    os.environ.pop('TTD_QWEN_SAMPLE_GROUP_CHUNK_SIZE', None)
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
    think_budget = completer._think_budget(ob.length)          # 22,000 unless FIT shrinks it for a long prompt
    answer_budget = completer._output_limit(ob.length) - think_budget
    settings = dict(model=MODEL, adapter=None, renderer='tml_v0', effort=EFFORT, temperature=TEMPERATURE,
                    completer='InklingTwoPhaseTokenCompleter', context_window=CONTEXT, phase1_max_tokens=PHASE1,
                    fixed_think_tokens=think, fixed_total_tokens=total, fixed_answer_tokens=total - think,
                    fixed_budget_fit=FIXED_FIT, min_think_tokens=0, answer_cue=completer.ANSWER_CUE,
                    this_prompt=dict(prompt_tokens=ob.length, think_budget=think_budget, answer_budget=answer_budget),
                    source='Inkling-2-Mini production recipe = mc-s5 / mn9-s5 env.txt', seed=None, key=a.key,
                    groups=a.groups, n=a.n, problem=a.problem, prompt_file=os.path.abspath(a.prompt), started=time.time())
    print(f'{a.problem}: prompt {ob.length} tokens, thinking {think_budget} + answer {answer_budget}, '
          f'{a.groups} x {a.n}', flush=True)

    async def group(g):
        t0 = time.time()
        seqs = await completer.sample_group(ob, stop, a.n)
        return g, t0, time.time(), seqs

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_rows = 0
    json.dump(settings, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'meta.json'), 'w'), indent=1)
    with open(a.out, 'w') as fh:
        fh.write(json.dumps(dict(header=True, settings=settings)) + '\n')
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
                           think_budget=think_budget, answer_budget=answer_budget, t_start=round(t0, 1),
                           t_end=round(t1, 1), seconds=round(t1 - t0, 1), key=a.key)
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
    p.add_argument('--think', type=int, default=FIXED_THINK, help='production 22,000; change only for tests')
    p.add_argument('--total', type=int, default=FIXED_TOTAL, help='production 32,000; change only for tests')
    asyncio.run(main(p.parse_args()))
