"""Paired, fixed-adapter 16x32 inference benchmark for an idle hybrid ingress.

This measures generation only. A passing result is not a training, grading,
checkpoint-resume, or full-step throughput gate.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import uuid

import httpx


def validate(response, request):
    from skyrl.backends.native_completion import validate_choice
    choices = response.get('choices', [])
    if len(choices) != request['n'] or {c.get('index') for c in choices} != set(range(request['n'])):
        raise ValueError('missing or duplicate completions')
    for choice in choices:
        validate_choice(copy.deepcopy(choice), request['thinking_token_budget'], request['max_tokens'])
        ids = choice['token_ids']
        probs = (choice.get('logprobs') or {}).get('token_logprobs')
        if not isinstance(probs, list) or len(probs) != len(ids):
            raise ValueError('missing sampled-token logprobs')
        if any(type(x) not in (float, int) or not math.isfinite(x) or x > 1e-5 for x in probs):
            raise ValueError('invalid sampled-token logprobs')
        if any(x not in (0, 1) for x in choice['loss_mask']):
            raise ValueError('invalid loss mask')
        if choice.get('finish_reason') not in ('stop', 'length'):
            raise ValueError('invalid completion finish reason')


def comparison(rounds):
    measured = [r for r in rounds if not r['warmup']]
    local = [r['seconds'] for r in measured if r['route'] == 'local']
    hybrid = [r['seconds'] for r in measured if r['route'] == 'hybrid']
    if len(local) < 3 or len(hybrid) != len(local):
        raise ValueError('need at least three paired measured rounds')
    a, b = statistics.median(local), statistics.median(hybrid)
    reduction = 1 - b / a
    return dict(local_median_seconds=a, hybrid_median_seconds=b,
                generation_time_reduction=reduction, generation_gate_passed=reduction >= .10,
                full_step_gate='not_measured')


async def run(args):
    requests = json.loads(args.requests.read_text())
    if (not isinstance(requests, list) or len(requests) != 16
            or any(r.get('n') != 32 or 'thinking_token_budget' not in r or r.get('logprobs') != 1
                   or not r.get('return_token_ids') for r in requests)):
        raise ValueError('expected 16 native-thinking payloads, n=32, token IDs and logprobs enabled')
    requests = [dict(r, model=args.model) for r in requests]
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    save('requests.json', requests)
    phase = 'benchmark-' + uuid.uuid4().hex
    async with httpx.AsyncClient(base_url=args.endpoint.rstrip('/'), timeout=args.timeout) as http:
        async def call(path, body=None, **kw):
            response = await (http.get(path, **kw) if body is None else http.post(path, json=body, **kw))
            response.raise_for_status()
            return response.json()
        before = await call('/status')
        if before.get('active') or before.get('updating') or before.get('borrowing', {}).get('phase'):
            raise ValueError('benchmark requires an idle, exclusively owned ingress with the training client paused')
        if before.get('committed') != args.model or 'scheduling' not in before:
            raise ValueError('benchmark requires a committed adapter and hybrid scheduler')
        save('before.json', before)
        await call('/skyrl/v1/borrowing/begin', dict(phase_id=phase, expected_n=32))
        rounds = []
        try:
            deadline = time.monotonic() + args.prepare_timeout
            while True:
                ready = await call('/status')
                if ready.get('borrowing', {}).get('ready'):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('farm did not finish same-adapter publication')
                await asyncio.sleep(2)
            digest = ready['borrowing']['adapter_sha256']
            if not digest:
                raise ValueError('benchmark cannot use an unadapted base model')
            save('ready.json', ready)
            # Warm both routes, then alternate AB/BA to reduce order bias.
            for pair in range(args.rounds + 1):
                routes = ('local', 'hybrid') if pair % 2 == 0 else ('hybrid', 'local')
                for route in routes:
                    round_before = await call('/status')
                    started = time.monotonic()
                    label = f'{pair:02d}-{route}'
                    async def one(index, payload):
                        t = time.monotonic()
                        response = await call('/v1/completions', payload,
                            headers={'X-SkyRL-Local-Only': '1' if route == 'local' else '0'})
                        save(f'{label}-{index:02d}.json', response)
                        validate(response, payload)
                        return dict(index=index, seconds=time.monotonic()-t,
                                    tokens=sum(len(c['token_ids']) for c in response['choices']))
                    groups = await asyncio.gather(*(one(i, r) for i, r in enumerate(requests)))
                    after = await call('/status')
                    save(label + '-status.json', after)
                    counts = {name: after['scheduling']['completed'][name] -
                                    round_before['scheduling']['completed'][name]
                              for name in ('local', 'remote')}
                    if sum(counts.values()) != 16 or (route == 'local' and counts['remote']) or (
                            route == 'hybrid' and not counts['remote']):
                        raise RuntimeError('benchmark routing counts do not prove the requested comparison')
                    if (after.get('committed') != args.model or after.get('starts') != before.get('starts')
                            or after.get('borrowing', {}).get('adapter_sha256') != digest
                            or not after.get('borrowing', {}).get('ready')):
                        raise RuntimeError('adapter, engine, or farm changed during paired benchmark')
                    record = dict(pair=pair, warmup=pair == 0, route=route,
                                  seconds=time.monotonic()-started, groups=groups, completed=counts)
                    rounds.append(record)
                    save('rounds.json', rounds)
                    print(json.dumps({k: v for k, v in record.items() if k != 'groups'}), flush=True)
            summary = comparison(rounds)
            summary.update(adapter_sha256=digest,
                requests_sha256=hashlib.sha256((args.output/'requests.json').read_bytes()).hexdigest())
            save('summary.json', summary)
        finally:
            await call('/skyrl/v1/borrowing/end', dict(phase_id=phase))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True, help='Current committed adapter alias')
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=21600)
    parser.add_argument('--prepare-timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.rounds < 3 or min(args.timeout, args.prepare_timeout) <= 0:
        parser.error('require at least three measured rounds and positive timeouts')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
