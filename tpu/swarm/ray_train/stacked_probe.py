"""Short real-sampling / stacked-backward probe; synthetic signed advantages.

This is a systems correctness probe, not an Erdős reward experiment. The full
profile runs the real RL client separately with 32 pooled rollouts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import sys
import time


def datum_from_sample(prompt, tokens, logprobs, advantage):
    import tinker
    if not tokens or logprobs is None or len(tokens) != len(logprobs):
        raise ValueError("probe requires sampled tokens and aligned behavior logprobs")
    if any(not math.isfinite(p) or p <= -9000 for p in logprobs):
        raise ValueError("probe received invalid behavior logprobs")
    sequence = prompt + tokens
    prefix = len(prompt) - 1
    def tensor(data, dtype='float32'):
        return tinker.TensorData(data=data, dtype=dtype, shape=[len(data)])
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(sequence[:-1]), loss_fn_inputs={
        'target_tokens': tensor(sequence[1:], 'int64'),
        'weights': tensor([1.] * (len(sequence) - 1)),
        'advantages': tensor([0.] * prefix + [advantage] * len(tokens)),
        'logprobs': tensor([0.] * prefix + logprobs),
    })


async def run(args):
    sys.path[:0] = [str(Path(args.source) / 'third_party/discover'), args.source]
    import tinker
    from transformers import AutoTokenizer
    from ttt_discover.rl.multi_lora import serialize_datum
    from ttt_discover.rl.multi_lora_request import submit_multi_lora
    output = Path(os.environ['TTD_RUN_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    def record(event, **fields):
        row = dict(event=event, time=time.time(), **fields)
        print(json.dumps(row), flush=True)
        with (output / 'stacked_probe.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    service = tinker.ServiceClient(base_url=os.environ['TINKER_BASE_URL'])
    clients = [await asyncio.to_thread(service.create_lora_training_client,
               base_model=args.model, rank=32, seed=seed, train_unembed=False) for seed in (1, 2)]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = tokenizer.encode('Continue this list of integers: 1, 2, 3, 4, 5,', add_special_tokens=True)
    assert len(prompt) + 64 <= 512
    ids = [c.model_id for c in clients]
    record('adapters_created', models=ids)
    for step in range(3):
        paths = [(await asyncio.to_thread(lambda c=c: c.save_weights_for_sampler(f'probe-{step}').result())).path
                 for c in clients]
        samplers = [await asyncio.to_thread(service.create_sampling_client, model_path=path) for path in paths]
        async def sample(index):
            response = await asyncio.to_thread(lambda: samplers[index].sample(
                tinker.ModelInput.from_ints(prompt), num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=64, temperature=1.)).result())
            return index, response.sequences[0]
        samples = await asyncio.gather(*(sample(i) for i in (0, 0, 1, 1)))
        record('sampled', step=step, versions=paths, lengths=[len(s.tokens) for _, s in samples])
        if step == 2:
            break  # Explicitly sample the adapters after both optimizer updates.
        data = [datum_from_sample(prompt, s.tokens, s.logprobs, 1. if row % 2 == 0 else -1.)
                for row, (_, s) in enumerate(samples)]
        (output / f'probe_batch_{step}.json').write_text(json.dumps(dict(
            datums=[serialize_datum(d) for d in data], owners=[ids[i] for i, _ in samples], versions=paths)))
        future = await submit_multi_lora(os.environ['TINKER_BASE_URL'], ids, data, 2., f'probe-{step}',
                                        [ids[i] for i, _ in samples], [paths[i] for i, _ in samples])
        result = await future.result_async()
        assert result['metrics']['stacked_adapter_execution'] == 1
        assert all(r['metrics']['stacked_replay_verified'] == 1 for r in result['results'].values())
        record('backward_verified', step=step, metrics=result['metrics'],
               adapter_metrics={name: r['metrics'] for name, r in result['results'].items()})
        for index, client in enumerate(clients):
            rate = 1e-4 * (index + 1)
            update = await asyncio.to_thread(lambda: client.optim_step(tinker.AdamParams(
                learning_rate=rate, beta1=.9, beta2=.95, eps=1e-8)).result())
            record('optimizer_step', step=step, adapter=ids[index], learning_rate=rate, metrics=update.metrics)
    checkpoints = [(await asyncio.to_thread(lambda c=c: c.save_state('probe-final').result())).path for c in clients]
    record('probe_complete', checkpoints=checkpoints, optimizer_updates=4, post_update_sampling=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', required=True)
    asyncio.run(run(parser.parse_args()))
