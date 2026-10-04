"""Cross-farm logprob parity on the base model, driven from the login node.

For each farm (an UNLEASED farm only: a farm held by a trainer refuses the
probe lease and is skipped), a stdlib script on the farm head takes a short
probe lease, sends the same greedy (temperature 0) completions to its ingress
on 127.0.0.1, and releases the lease:

  sequential  every prompt alone, `--repeats` times (batch of one)
  batched     every prompt at once (shares a batch with the others)

Greedy outputs from two engines are the same token sequence until they first
disagree, and up to that point both report the logprob of the same token in
the same context, so the common prefix is an exact teacher-forced
comparison. vLLM TPU cannot score a given sequence (prompt_logprobs kills its
EngineCore), so this tool never sends prompt_logprobs.

Comparisons (per prompt, over the common prefix):
  repeat    same farm, sequential run 1 vs run 2     nondeterminism floor
  batch     same farm, sequential vs batched         batch-composition effect
  cross     farm A vs farm B, sequential run 1       hardware / TP effect

Trainer vs farm numerics are measured online on real rollouts by the
`sampler_mismatch/*` training metrics (third_party/discover/.../sampler_mismatch.py).

Example:
  python -m tpu.swarm.ray_train.logprob_parity \\
      --ssh-dir .../.sky/generated/ssh --farm <cluster-a> --farm <cluster-b> \\
      --out parity.json
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

REMOTE = r'''
import json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
P = PARAMS
def call(method, path, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (P['port'], path), data=data, method=method,
        headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {'raw': e.read().decode(errors='replace')[:400]}
out = dict(started=time.time())
code, status = call('GET', '/status')
caps = status.get('capabilities') or {}
out['state'] = status.get('state')
out['compatibility_sha256'] = caps.get('compatibility_sha256')
code, models = call('GET', '/v1/models')
versions = set(status.get('versions') or [])
names = [m['id'] for m in models.get('data', []) if m['id'] not in versions]
model = P.get('model') or (names[0] if names else None)
out['model'] = model
code, ack = call('POST', '/acquire_lease', {'owner_run': P['owner'], 'ttl_seconds': P['ttl'],
                                           'compatibility_sha256': caps.get('compatibility_sha256')})
if code != 200:
    out['error'] = 'lease refused (%s): %s' % (code, json.dumps(ack)[:300])
    print(json.dumps(out)); raise SystemExit
H = {'X-Lease-ID': ack['lease_id']}
def complete(prompt):
    body = {'model': model, 'prompt': prompt['prompt'], 'max_tokens': P['max_tokens'], 'temperature': 0.0,
            'n': 1, 'logprobs': 1, 'stream': False, 'return_token_ids': True}
    t0 = time.time()
    code, r = call('POST', '/v1/completions', body, headers=H, timeout=P['request_timeout'])
    rec = dict(id=prompt['id'], status=code, seconds=round(time.time() - t0, 3))
    if code == 200 and r.get('choices'):
        c = r['choices'][0]
        rec.update(tokens=c.get('token_ids') or [], logprobs=(c.get('logprobs') or {}).get('token_logprobs') or [],
                   finish_reason=c.get('finish_reason'), served_by=c.get('served_by'))
    else:
        rec['error'] = json.dumps(r)[:300]
    return rec
try:
    out['sequential'] = [[complete(p) for p in P['prompts']] for _ in range(P['repeats'])]
    with ThreadPoolExecutor(max_workers=len(P['prompts'])) as pool:
        out['batched'] = list(pool.map(complete, P['prompts']))
finally:
    out['release'] = call('POST', '/release_lease', {'lease_id': ack['lease_id']}, headers=H)[0]
print(json.dumps(out))
'''

PROMPTS = [
    ("ac2", "Find a nonnegative step function f on [-1/4, 1/4] that minimizes max_t (f*f)(t) / (integral of f)^2. "
            "Write a Python function construct_function() returning the step heights as a list."),
    ("proof", "Prove that there are infinitely many primes congruent to 3 mod 4."),
    ("code", "Write an efficient Python function that returns the longest palindromic substring of a string, "
             "then explain its complexity."),
    ("numeric", "Compute 37 * 43 - 19 * 23 step by step, then check the result a second way."),
    ("prose", "Describe how a suspension bridge distributes load, for a curious high school student."),
    ("list", "List the first twenty Fibonacci numbers and state which of them are prime."),
    ("algebra", "Solve x^3 - 6x^2 + 11x - 6 = 0 and verify each root."),
    ("summary", "Summarize the trade-offs between data parallelism, tensor parallelism and pipeline parallelism."),
]


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def compare(a, b):
    """Teacher-forced comparison of two greedy records over their common prefix."""
    ta, tb = list(a.get('tokens') or []), list(b.get('tokens') or [])
    la, lb = a.get('logprobs') or [], b.get('logprobs') or []
    shortest = min(len(ta), len(tb), len(la), len(lb))
    prefix = next((i for i in range(shortest) if ta[i] != tb[i]), shortest)
    xa = np.asarray([np.nan if v is None else v for v in la[:prefix]], dtype=np.float64)
    xb = np.asarray([np.nan if v is None else v for v in lb[:prefix]], dtype=np.float64)
    keep = np.isfinite(xa) & np.isfinite(xb)
    diffs = xa[keep] - xb[keep]
    return dict(identical=ta == tb, prefix=int(prefix), shortest=int(min(len(ta), len(tb))),
                diverged_at=int(prefix) if prefix < min(len(ta), len(tb)) else None,
                diffs=diffs.tolist())


def summarize(comparisons):
    diffs = np.asarray([d for c in comparisons for d in c['diffs']], dtype=np.float64)
    absd = np.abs(diffs)
    fractions = [c['prefix'] / c['shortest'] for c in comparisons if c['shortest']]
    return dict(pairs=len(comparisons), identical=sum(c['identical'] for c in comparisons),
                tokens=int(diffs.size),
                mean_abs=float(absd.mean()) if diffs.size else None,
                mean_diff=float(diffs.mean()) if diffs.size else None,
                p99_abs=float(np.quantile(absd, 0.99)) if diffs.size else None,
                max_abs=float(absd.max()) if diffs.size else None,
                frac_abs_ge_0p01=float((absd >= 0.01).mean()) if diffs.size else None,
                median_prefix_fraction=float(np.median(fractions)) if fractions else None)


def analyze(results):
    """results: {farm label: remote output}. Returns {kind or 'cross a|b': summary}."""
    def by_id(records):
        return {r['id']: r for r in records if r.get('tokens')}
    report = {}
    usable = {k: v for k, v in results.items() if v.get('sequential')}
    for farm, out in usable.items():
        runs = [by_id(run) for run in out['sequential']]
        if len(runs) > 1:
            report[f'repeat {farm}'] = summarize([compare(runs[0][i], runs[1][i])
                                                  for i in runs[0] if i in runs[1]])
        batched = by_id(out.get('batched') or [])
        report[f'batch {farm}'] = summarize([compare(runs[0][i], batched[i]) for i in runs[0] if i in batched])
    farms = sorted(usable)
    for x, farm_a in enumerate(farms):
        for farm_b in farms[x + 1:]:
            a, b = by_id(usable[farm_a]['sequential'][0]), by_id(usable[farm_b]['sequential'][0])
            report[f'cross {farm_a} | {farm_b}'] = summarize([compare(a[i], b[i]) for i in a if i in b])
    return report


def label(cluster, out):
    for run in out.get('sequential') or []:
        for record in run:
            stamp = record.get('served_by')
            if stamp:
                return f"{stamp.get('accelerator')}-tp{stamp.get('tp')} ({cluster})"
    return cluster


def run_farm(args, cluster, prompts):
    params = dict(port=args.port, owner=f'logprob-parity-{int(time.time())}', ttl=args.ttl,
                  max_tokens=args.max_tokens, repeats=args.repeats, request_timeout=args.request_timeout,
                  model=args.model, prompts=prompts)
    config = Path(args.ssh_dir) / cluster
    proc = subprocess.run(['ssh', '-F', str(config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', cluster,
                           'python3 -'], input='PARAMS = ' + repr(params) + '\n' + REMOTE, text=True,
                          capture_output=True, timeout=args.ttl + 120)
    if proc.returncode:
        return dict(error=f'ssh failed ({proc.returncode}): {proc.stderr[-400:]}')
    return json.loads(proc.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--farm', action='append', required=True, help='farm head cluster name (repeat)')
    parser.add_argument('--ssh-dir', required=True)
    parser.add_argument('--port', type=int, default=24800)
    parser.add_argument('--model', default='', help='served model id; the base model when omitted')
    parser.add_argument('--prompts', help='JSONL with {"id", "prompt"} (raw text, sent as is)')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--ttl', type=int, default=900, help='probe lease seconds (also bounds the run)')
    parser.add_argument('--request-timeout', type=int, default=600)
    parser.add_argument('--out', default='logprob-parity.json')
    args = parser.parse_args()
    if args.prompts:
        prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    else:
        prompts = [dict(id=key, prompt=chat(text)) for key, text in PROMPTS]
    results = {}
    for cluster in args.farm:
        out = run_farm(args, cluster, prompts)
        if out.get('error'):
            print(f'{cluster}: {out["error"]}', flush=True)
        results[label(cluster, out)] = out
    report = analyze(results)
    Path(args.out).write_text(json.dumps(dict(results=results, report=report), indent=1))
    print(f'{"comparison":<60} {"pairs":>5} {"same":>5} {"tokens":>7} {"mean|d|":>9} {"p99|d|":>9} '
          f'{"max|d|":>9} {"prefix":>7}')
    for name, s in report.items():
        fmt = lambda v: f'{v:9.2e}' if isinstance(v, float) else f'{"-":>9}'
        prefix = s['median_prefix_fraction']
        print(f'{name:<60} {s["pairs"]:>5} {s["identical"]:>5} {s["tokens"]:>7} {fmt(s["mean_abs"])} '
              f'{fmt(s["p99_abs"])} {fmt(s["max_abs"])} {prefix if prefix is None else round(prefix, 3):>7}')
    print(f'raw records and report: {args.out}')


if __name__ == '__main__':
    main()
