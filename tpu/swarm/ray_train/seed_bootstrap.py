"""Bounded, frozen-model bootstrap using the same environments as GRPO.

Only generation and grading run here. The controller starts the trainer after
this journal and its independent seed roots have been durably promoted.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

TASKS = {
    'ac_inequalities': ('examples.ac_inequalities.env', 'AutoCorrInequalityEnv'),
    'circle_packing': ('examples.circle_packing.env', 'CirclePackingEnv'),
    'science_routing': ('tpu.science.training_env', 'RoutingTrainingEnv'),
    'science_placement': ('tpu.science.training_env', 'PlacementTrainingEnv'),
    'recurrent_gemma': ('pallas_arena.rl.env', 'RecurrentGemmaEnv'),
}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    from ttt_discover.tinker_utils.state import to_json_serializable
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(to_json_serializable(value), stream, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def environment_type(config):
    module, name = TASKS[config.client_env['TTD_ENV']]
    return getattr(importlib.import_module(module), name)


def make_environment(config, state, renderer, path):
    from ttt_discover.tinker_utils.dataset_builder import DatasetConfig
    cls = environment_type(config)
    timeout = int(os.environ.get('EVAL_TIMEOUT', config.client_env.get('EVAL_TIMEOUT', '1100')))
    settings = DatasetConfig(problem_type=config.client_env['TTD_PROBLEM_TYPE'], env_type=cls,
        batch_size=16, model_name_for_tokenizer=config.model,
        renderer_name=config.client_member_spec.split(':')[1], group_size=32,
        num_cpus_per_task=int(config.client_env.get('NUM_CPUS_PER_TASK', '2')),
        eval_backend=os.environ.get('TTD_EVAL_BACKEND', config.client_env.get('TTD_EVAL_BACKEND', 'ray')),
        eval_timeout=timeout, timeout=max(8000, timeout + 60), log_path=str(path))
    return cls(renderer=renderer, initial_state=state, sampler=object(), config=settings)


def selected_states(rows, limit):
    """Rank valid states, deduplicate source, and bypass training's sibling cap."""
    candidates = [r for r in rows if r['correctness'] == 1 and r.get('state')
                  and r['state'].get('code') and math.isfinite(r['state']['value'])]
    candidates.sort(key=lambda r: (-r['state']['value'], r['id']))
    seen, selected = set(), []
    for row in candidates:
        key = row['state']['code'].strip()
        if key in seen: continue
        seen.add(key)
        state = dict(row['state'], parents=[], parent_values=[])
        selected.append(state)
        if len(selected) == limit: break
    return selected


def outstanding_limit(config, valid_count):
    remaining = max(0, config.bootstrap_target_valid - valid_count)
    return min(config.bootstrap_max_groups, math.ceil(remaining / config.bootstrap_group_size))


async def collect(config, folder, generate, grade):
    """Journal full response batches before grading; resume only missing work."""
    folder = Path(folder)
    limit = config.bootstrap_max_drafts // config.bootstrap_group_size
    rows, completed, launched = [], set(), set()
    def saved_grade(path, group, sample):
        row = read(path)
        if row.get('id') != f'bootstrap-{group:03d}-{sample:03d}':
            raise ValueError('bootstrap grade journal identity mismatch')
        return row

    for group_dir in sorted(folder.glob('group-*')):
        index = int(group_dir.name.split('-')[-1])
        if not 0 <= index < limit: raise ValueError('bootstrap journal exceeds configured cap')
        launched.add(index)
        generation = group_dir / 'generation.json'
        if not generation.exists():
            if list(group_dir.glob('grade-*.json')):
                raise ValueError('bootstrap journal missing generation for saved grades')
            continue
        choices = read(generation)['choices']
        if len(choices) != config.bootstrap_group_size:
            raise ValueError('bootstrap journal has incomplete generation batch')
        grades = list(group_dir.glob('grade-*.json'))
        for path in sorted(grades):
            sample = int(path.stem.split('-')[-1])
            if not 0 <= sample < config.bootstrap_group_size:
                raise ValueError('bootstrap grade journal index exceeds group size')
            rows.append(saved_grade(path, index, sample))
        if len(grades) == config.bootstrap_group_size:
            completed.add(index)

    async def group(index):
        directory = folder / f'group-{index:03d}'
        generation = directory / 'generation.json'
        if generation.exists():
            choices = read(generation)['choices']
        else:
            choices = await generate(index)
            if len(choices) != config.bootstrap_group_size:
                raise RuntimeError('incomplete bootstrap generation batch')
            save(generation, dict(choices=choices))
        async def one(i, choice):
            path = directory / f'grade-{i:03d}.json'
            if path.exists(): return saved_grade(path, index, i)
            row = await grade(index, i, choice)
            save(path, row)
            return row
        batch = await asyncio.gather(*(one(i, c) for i, c in enumerate(choices)))
        print(json.dumps(dict(event='bootstrap_group_complete', group=index,
            valid=sum(r['correctness'] == 1 for r in batch), total=len(batch))), flush=True)
        return index, batch

    pending = {}
    try:
        while True:
            by_id = {r['id']: r for r in rows}
            count = len(selected_states(list(by_id.values()), config.bootstrap_target_valid))
            desired = outstanding_limit(config, count)
            while len(pending) < desired:
                index = next((i for i in range(limit) if i not in completed and i not in pending), None)
                if index is None: break
                launched.add(index)
                pending[index] = asyncio.create_task(group(index))
            if not pending: break
            done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index, batch = await task
                rows.extend(batch); completed.add(index); del pending[index]
            # Existing groups are allowed to finish; no fresh groups after target.
        return list({r['id']: r for r in rows}.values()), len(launched) * config.bootstrap_group_size
    finally:
        for task in pending.values(): task.cancel()
        await asyncio.gather(*pending.values(), return_exceptions=True)


async def run(config, snapshot, head):
    import httpx
    import ray
    from transformers import AutoTokenizer
    from ttt_discover.tinker_utils import renderers
    from ttt_discover.tinker_utils.dataset_builder import answer_only_code
    from .frozen_benchmark import payload, check_choice

    config.validate()
    if not config.bootstrap_max_drafts: raise ValueError('bounded bootstrap is disabled')
    if not ray.is_initialized():
        ray.init(address=os.environ['RAY_ADDRESS'], namespace=os.environ['RAY_NAMESPACE'])
    folder = Path(os.environ['TTD_RUN_DIR']) / 'bootstrap'
    log = Path(os.environ['TTD_RUN_DIR']) / 'tinker_log' / config.run_id
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    renderer = renderers.get_renderer(config.client_member_spec.split(':')[1], tokenizer=tokenizer)
    cls = environment_type(config)
    root_path = folder / 'root.json'
    if root_path.exists(): root = cls.state_type.from_dict(read(root_path))
    else:
        root = cls.create_initial_state(config.client_env['TTD_PROBLEM_TYPE'])
        save(root_path, root.to_dict())
    env = make_environment(config, root, renderer, folder)
    question = env.get_question()
    rendered = renderer.build_generation_prompt([{'role': 'user', 'content': question}])
    request = payload(config, SimpleNamespace(prompt=rendered.to_ints(), stop_ids=renderer.get_stop_sequences()),
                      n=config.bootstrap_group_size)
    contract = dict(schema=2, config=config.to_dict(), prompt=question,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy='frozen-base-no-adapter', tokenizer=identity(tokenizer.get_vocab()),
        request_sha256=identity(request), root_sha256=identity(root.to_dict()))
    fingerprint = identity(contract)
    if (folder / 'contract.json').exists():
        if read(folder / 'contract.json')['sha256'] != fingerprint:
            raise RuntimeError('bootstrap contract changed; use a new run ID')
    else:
        if list(log.glob('puct_sampler_step_*.json')) or list(log.glob('member_*/checkpoints.jsonl')):
            raise RuntimeError('refusing to bootstrap an existing training history')
        save(folder / 'contract.json', dict(contract=contract, sha256=fingerprint))
    if (folder / 'complete.json').exists(): return read(folder / 'complete.json')
    family = config.client_env['TTD_ANSWER_MODEL_FAMILY']
    async with httpx.AsyncClient(timeout=config.inference.request_timeout) as http:
        async def generate(index):
            response = await http.post(f'http://{head}:{config.ports.inference}/v1/completions', json=request)
            response.raise_for_status()
            choices = response.json()['choices']
            for choice in choices: check_choice(choice, request)
            return choices
        async def grade(group, index, choice):
            check_choice(choice, request)
            e = make_environment(config, root, renderer, folder / f'group-{group:03d}' / f'eval-{index:03d}')
            message, parsed = renderer.parse_response(choice['token_ids'])
            code = answer_only_code(message, family, e._should_keep_code_separators())
            identifier = f'bootstrap-{group:03d}-{index:03d}'
            row = dict(id=identifier, correctness=0, reward=0., code=code, state=None,
                       message='Invalid answer format or truncated answer', root_id=root.id)
            if code and choice.get('finish_reason') != 'length':
                verdict = await e._safe_grade(code, 0)
                if not math.isfinite(verdict.reward) or not math.isfinite(verdict.raw_score):
                    row['message'] = 'Nonfinite grading result'; return row
                state = e._create_next_state(0, code, verdict)
                state.id = identifier; state.origin = config.client_member_spec.split(':')[-1]
                row.update(correctness=verdict.correctness, reward=verdict.reward, message=verdict.msg,
                           metrics=verdict.metrics, state=state.to_dict() if verdict.correctness == 1 else None)
            return row
        rows, drafted = await collect(config, folder / 'drafts', generate, grade)
    selected = selected_states(rows, config.bootstrap_target_valid)
    summary = dict(contract_sha256=fingerprint, total=len(rows), drafted=drafted,
        valid=sum(r['correctness'] == 1 for r in rows), retained=len(selected), layers=1,
        optimizer_steps=0, stop_reason='valid_target' if len(selected) >= config.bootstrap_target_valid else 'draft_cap')
    if not selected:
        save(folder / 'no-valid-seeds.json', summary)
        raise RuntimeError('bootstrap produced no valid seeds; refusing optimizer startup')
    # Roots are not siblings and not permanent initial states. Normal GRPO top-2
    # admission and the existing 1000-state buffer cap apply to future children.
    pool = dict(step=0, states=selected, initial_states=[], puct_n={}, puct_m={}, puct_T=0)
    if any(p.name != 'puct_sampler_step_000000.json' for p in log.glob('puct_sampler_step_*.json')):
        raise RuntimeError('refusing to overwrite an optimizer-era state pool')
    save(log / 'puct_sampler_step_000000.json', pool)
    summary['pool_sha256'] = identity(pool)
    save(folder / 'complete.json', summary)
    print(json.dumps(dict(event='bootstrap_complete', **summary)), flush=True)
    return summary


def main():
    from . import config as config_module
    from .config import Config
    parser = argparse.ArgumentParser()
    for key in ('config-json', 'snapshot', 'head'): parser.add_argument('--' + key, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); config = Config.from_dict(json.loads(args.config_json))
    if Path(config_module.__file__).resolve() != Path(__file__).resolve().with_name('config.py'):
        raise RuntimeError('bootstrap/config package mismatch')
    config.validate()
    if args.check_only:
        cls = environment_type(config)
        root = cls.create_initial_state(config.client_env['TTD_PROBLEM_TYPE'])
        e = make_environment(config, root, None, Path('/tmp/bootstrap-import-check'))
        if not e.get_question().strip(): raise RuntimeError('empty bootstrap prompt')
        print(json.dumps(dict(event='bootstrap_import_check_passed', layers=config.bootstrap_layers)))
        return
    asyncio.run(run(config, args.snapshot, args.head))


if __name__ == '__main__': main()
