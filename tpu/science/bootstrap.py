"""Frozen base-policy seed discovery, isolated from every gradient/optimizer API.

The second layer repairs invalid programs only. Durable generation and per-code
grading records permit restart without redrawing completed groups. Failed repair
parents live only in this journal; successful children enter PUCT under the
original task root, with the actual repair-parent ID retained in the journal.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def repair_parents(rows, count, fits):
    """Select only invalid, distinct programs; spread across error signatures."""
    buckets, seen = {}, set()
    for row in rows:
        code = row['code']
        if row['correctness'] == 1 or not code or code in seen or not fits(row):
            continue
        seen.add(code)
        signature = row['message'].splitlines()[-1][:160]
        buckets.setdefault(signature, []).append(row)
    selected = []
    while buckets and len(selected) < count:
        for key in list(buckets):
            selected.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
            if len(selected) == count:
                break
    # Fewer than count eligible failures means fewer repair groups, never repairs
    # of valid programs or new drafts disguised as refinement.
    return selected


def commit_rows(sampler, roots, rows):
    """Existing top-k/dedup admission and one visit per graded candidate."""
    from ttt_discover import State
    by_root = {}
    for row in rows:
        parent = roots[row['root_id']]
        if row['correctness'] == 1:
            child = State(timestep=-2-row['layer'], construction=None,
                          code=row['code'], value=row['reward'], id=row['id'],
                          observation=row['feedback'], origin=row['origin'])
            by_root.setdefault(parent.id, []).append(child)
        else:
            sampler.record_failed_rollout(parent)
    for rid, children in by_root.items():
        parent = roots[rid]
        sampler.update_states(children, [parent]*len(children), save=False)
        for _ in children[1:]:
            sampler.record_failed_rollout(parent)
    sampler.flush(step=0)


async def run(config, snapshot, head):
    import httpx
    from transformers import AutoTokenizer
    from ttt_discover import State
    from ttt_discover.tinker_utils import renderers
    from ttt_discover.tinker_utils.dataset_builder import answer_only_code
    from ttt_discover.tinker_utils.sampler import PUCTSampler
    from tpu.swarm.ray_train.frozen_benchmark import payload, check_choice
    from .training_env import (ScienceTrainingEnv, RoutingTrainingEnv,
                              PlacementTrainingEnv, candidate_prompt, connect)

    config.validate()
    if not config.bootstrap_layers:
        raise ValueError('bootstrap is disabled')
    connect()
    task = config.science_task
    env_type = RoutingTrainingEnv if task == 'routing' else PlacementTrainingEnv
    groups = int(config.client_env.get('GROUPS_PER_BATCH', '16'))
    size = int(config.client_env.get('GROUP_SIZE', '32'))
    family = config.client_env['TTD_ANSWER_MODEL_FAMILY']
    folder = Path(os.environ['TTD_RUN_DIR']) / 'bootstrap'
    log = Path(os.environ['TTD_RUN_DIR']) / 'tinker_log' / config.run_id
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    renderer = renderers.get_renderer(config.client_member_spec.split(':')[1], tokenizer=tokenizer)

    def request(question):
        rendered = renderer.build_generation_prompt([{'role': 'user', 'content': question}])
        return payload(config, SimpleNamespace(prompt=rendered.to_ints(),
                                               stop_ids=renderer.get_stop_sequences()), n=size)

    def question(row):
        return candidate_prompt(task, row['code'], row['feedback'], repair=True)

    contract = dict(schema=1, config=config.to_dict(), prompt=candidate_prompt(task),
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    policy='frozen-base-no-adapter', repair='invalid-only', groups=groups, size=size,
                    tokenizer=identity(tokenizer.get_vocab()))
    fingerprint = identity(contract)
    if (folder/'contract.json').exists():
        if read(folder/'contract.json')['sha256'] != fingerprint:
            raise ValueError('bootstrap resume contract changed; use a new run ID')
    else:
        if list(log.glob('puct_sampler_step_*.json')) or list(log.glob('member_*/checkpoints.jsonl')):
            raise ValueError('cannot bootstrap an existing training history; use a new run ID')
        save(folder/'contract.json', dict(sha256=fingerprint, contract=contract))
    if (folder/'complete.json').exists():
        return read(folder/'complete.json')

    if not (folder/'roots.json').exists():
        save(folder/'roots.json', [env_type.create_initial_state(task).to_dict() for _ in range(groups)])
    roots = {s.id: s for s in map(State.from_dict, read(folder/'roots.json'))}
    initial = [dict(root_id=s.id, question=candidate_prompt(task), repair_parent_id=None)
               for s in roots.values()]
    rows = []
    grade_slots = asyncio.Semaphore(max(64, config.hosts * config.science_routing_slots_per_host)
                                   if task == 'routing' else 64)
    base = f'http://{head}:{config.ports.inference}'
    async with httpx.AsyncClient(timeout=config.inference.request_timeout) as http:
        for layer in range(config.bootstrap_layers):
            plan_path = folder/f'layer-{layer}-plan.json'
            if plan_path.exists():
                plan = read(plan_path)
            elif layer == 0:
                plan = initial
                save(plan_path, plan)
            else:
                def fits(row):
                    try:
                        request(question(row))
                        return True
                    except ValueError:
                        return False
                parents = repair_parents(rows, groups, fits)
                plan = [dict(root_id=r['root_id'], question=question(r), repair_parent_id=r['id'])
                        for r in parents]
                save(plan_path, plan)

            async def group(index, parent):
                directory = folder/f'layer-{layer}'/f'group-{index:03d}'
                generation = directory/'generation.json'
                req = request(parent['question'])
                if generation.exists():
                    response = read(generation)
                else:
                    started = time.monotonic()
                    result = await http.post(base+'/v1/completions', json=req)
                    result.raise_for_status()
                    response = result.json()
                    if len(response.get('choices', [])) != size:
                        raise RuntimeError('incomplete bootstrap generation group')
                    for choice in response['choices']:
                        check_choice(choice, req)
                    save(directory/'request-summary.json', dict(
                        prompt_tokens=len(req['prompt']), thinking_token_budget=req['thinking_token_budget'],
                        max_tokens=req['max_tokens'], generation_seconds=time.monotonic()-started,
                        request_sha256=identity(req)))
                    save(generation, response)
                async def grade(index, choice):
                    path = directory/f'grade-{index:03d}.json'
                    if path.exists():
                        return read(path)
                    check_choice(choice, req)
                    message, parsed = renderer.parse_response(choice['token_ids'])
                    code = answer_only_code(message, family, False)
                    error = ('Truncated answer: return a complete fenced Python program.'
                             if choice.get('finish_reason') == 'length' else
                             'Invalid answer format: return one complete fenced Python program.')
                    reward, correctness, metrics, feedback = 0., 0., {}, error
                    # Match Environment.step: parser success is diagnostic;
                    # complete answer code is independently graded. Truncated
                    # generations retain the normal training rejection rule.
                    if code and choice.get('finish_reason') != 'length':
                        e = object.__new__(ScienceTrainingEnv)
                        e.problem_type, e.log_path = task, str(directory)
                        e.num_cpus_per_task, e.eval_backend = 4, 'local'
                        e.eval_timeout = int(config.client_env['EVAL_TIMEOUT'])
                        async with grade_slots:
                            verdict = await e._safe_grade(code, -2-layer)
                        reward, correctness, metrics = verdict.reward, verdict.correctness, verdict.metrics
                        error, feedback = verdict.msg, verdict.stdout
                    row = dict(id=f'bootstrap-{layer}-{parent["root_id"]}-{index}-{directory.name}',
                               root_id=parent['root_id'], repair_parent_id=parent['repair_parent_id'],
                               origin=config.client_member_spec.split(':')[-1], layer=layer,
                               format=bool(code and parsed),
                               code=code, reward=reward, correctness=correctness,
                               message=error, feedback=feedback, metrics=metrics)
                    save(path, row)
                    return row
                result = await asyncio.gather(*(grade(i, c) for i, c in enumerate(response['choices'])))
                print(json.dumps(dict(event='bootstrap_group_complete', layer=layer, group=index,
                                      valid=sum(r['correctness']==1 for r in result), total=len(result))), flush=True)
                return result

            completed = await asyncio.gather(*(group(i, p) for i, p in enumerate(plan)))
            rows.extend(r for batch in completed for r in batch)
            save(folder/f'layer-{layer}-summary.json', dict(
                groups=len(plan), total=sum(map(len, completed)),
                valid=sum(r['correctness']==1 for batch in completed for r in batch)))

    # Rebuild deterministically from journal, so a crash during pool promotion
    # cannot double-count visits or overwrite a later optimizer snapshot.
    private = folder/'pool'
    save(private/'puct_sampler_step_000000.json', dict(
        step=0, states=[s.to_dict() for s in roots.values()],
        initial_states=[s.to_dict() for s in roots.values()], puct_n={}, puct_m={}, puct_T=0))
    sampler = PUCTSampler(str(private/'puct_sampler.json'), env_type=env_type,
                          problem_type=task, batch_size=groups, resume_step=0)
    commit_rows(sampler, roots, rows)
    pool = read(private/'puct_sampler_step_000000.json')
    if any(p.name != 'puct_sampler_step_000000.json' for p in log.glob('puct_sampler_step_*.json')):
        raise RuntimeError('refusing to overwrite an optimizer-era state pool')
    save(log/'puct_sampler_step_000000.json', pool)
    summary = dict(contract_sha256=fingerprint, total=len(rows), valid=sum(r['correctness']==1 for r in rows),
                   retained=sum(bool(s['code']) for s in pool['states']), optimizer_steps=0,
                   pool_sha256=identity(pool), layers=config.bootstrap_layers)
    if summary['retained'] < 1:
        save(folder/'no-valid-seeds.json', summary)
        raise RuntimeError('bootstrap produced no valid seeds; refusing to start optimizer training')
    save(folder/'complete.json', summary)
    print(json.dumps(dict(event='bootstrap_complete', **summary)), flush=True)
    return summary


def main():
    import tpu.swarm.ray_train.config as config_module
    from tpu.swarm.ray_train.config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-json', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    expected = Path(__file__).resolve().parents[2] / 'tpu/swarm/ray_train/config.py'
    if Path(config_module.__file__).resolve() != expected:
        raise RuntimeError(f'bootstrap/config package mismatch: {__file__} vs {config_module.__file__}')
    config = Config.from_dict(json.loads(args.config_json))
    config.validate()
    if args.check_only:
        from tpu.swarm.ray_train.frozen_benchmark import payload, check_choice
        from .training_env import ScienceTrainingEnv, candidate_prompt
        from ttt_discover.tinker_utils import renderers
        from ttt_discover.tinker_utils.dataset_builder import answer_only_code
        from ttt_discover.tinker_utils.sampler import PUCTSampler
        print(json.dumps(dict(event='bootstrap_import_check_passed',
                              bootstrap=__file__, config=config_module.__file__,
                              layers=config.bootstrap_layers)), flush=True)
        return
    if config.borrows_inference:
        from tpu.swarm.ray_train.borrowing_phase import sampling_phase
        async def borrowed_run():
            async with sampling_phase(f'http://{args.head}:{config.ports.inference}', bootstrap=True):
                return await run(config, args.snapshot, args.head)
        asyncio.run(borrowed_run())
    else:
        asyncio.run(run(config, args.snapshot, args.head))


if __name__ == '__main__':
    main()
