"""Replay recorded Gemma failures at both trainer buckets, then test writeback.

Uses unchanged sampled tokens, advantages, masks, and behavior logprobs. This is
an attention/compiler regression probe, not a fresh RL quality experiment.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def restore_datum(raw, tinker):
    chunks = raw['model_input']['chunks']
    if any(c['type'] != 'encoded_text' for c in chunks):
        raise ValueError('replay fixture must contain encoded text only')
    tokens = [token for c in chunks for token in c['tokens']]
    inputs = {}
    for key, value in raw['loss_fn_inputs'].items():
        data = value['data']
        if data and len(data) != len(tokens):
            raise ValueError(f'unaligned fixture tensor: {key}')
        inputs[key] = tinker.TensorData(data=data, dtype='int64' if key == 'target_tokens' else 'float32',
                                        shape=[len(data)])
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(tokens), loss_fn_inputs=inputs)


def finite_metrics(metrics):
    if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in metrics.values()):
        raise ValueError(f'nonfinite metrics: {metrics}')


def run(args):
    sys.path[:0] = [str(Path(args.source) / 'third_party/discover'), args.source]
    import tinker
    fixture = json.loads(Path(__file__).with_name(getattr(args, 'fixture', 'gemma_attention_replay.json')).read_text())
    if fixture.get('model', args.model) != args.model:
        raise ValueError('replay fixture belongs to a different model')
    output = Path(os.environ['TTD_RUN_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    def record(event, **fields):
        row = dict(event=event, time=time.time(), **fields)
        print(json.dumps(row), flush=True)
        with (output / 'attention_replay.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    service = tinker.ServiceClient(base_url=os.environ['TINKER_BASE_URL'])
    client = service.create_lora_training_client(base_model=args.model, rank=32, seed=1, train_unembed=False)
    record('replay_started', source_run=fixture['source_run'], model_id=client.model_id)
    failures = []
    for case in fixture['cases']:
        raw_datums = case['datums'] if 'datums' in case else [case['datum']]
        datums = [restore_datum(raw, tinker) for raw in raw_datums]
        for repeat in range(2):
            started = time.monotonic()
            record('backward_started', bucket=case['bucket'], tokens=case['length'], repeat=repeat,
                   source_request=case['request_id'], datum_index=case.get('datum_index'),
                   examples=len(datums))
            try:
                result = client.forward_backward(datums, loss_fn=case['loss_fn'],
                                                 loss_fn_config=case['loss_fn_config']).result()
                finite_metrics(result.metrics)
                record('backward_completed', bucket=case['bucket'], repeat=repeat,
                       seconds=time.monotonic()-started, metrics=result.metrics)
            except Exception as exc:
                failures.append(dict(bucket=case['bucket'], error=str(exc)))
                record('backward_failed', bucket=case['bucket'], seconds=time.monotonic()-started, error=str(exc))
                break
    if failures:
        raise RuntimeError(f'Replay failed; no optimizer update submitted: {failures}')
    started = time.monotonic()
    update = client.optim_step(tinker.AdamParams(learning_rate=args.learning_rate, beta1=.9, beta2=.95, eps=1e-8)).result()
    finite_metrics(update.metrics)
    if update.metrics.get('skyrl.ai/grad_norm', 0) <= 0 or update.metrics.get('skyrl.ai/skipped_nonfinite_update', 0):
        raise RuntimeError(f'Optimizer did not apply a finite nonzero gradient: {update.metrics}')
    record('optimizer_completed', seconds=time.monotonic()-started, metrics=update.metrics)
    path = client.save_weights_for_sampler('replay-updated').result().path
    record('adapter_exported', path=path)
    sampler = service.create_sampling_client(model_path=path)
    # Reuse the genuine task prompt, stopping before its first trained token.
    first = fixture['cases'][0]
    raw = first['datums'][0] if 'datums' in first else first['datum']
    advantages = raw['loss_fn_inputs']['advantages']['data']
    prefix = next(i for i, value in enumerate(advantages) if value != 0) + 1
    tokens = [t for c in raw['model_input']['chunks'] for t in c['tokens']][:prefix]
    started = time.monotonic()
    result = sampler.sample(tinker.ModelInput.from_ints(tokens), num_samples=1,
                            sampling_params=tinker.SamplingParams(max_tokens=64, temperature=1.)).result()
    if len(result.sequences) != 1 or not result.sequences[0].tokens:
        raise RuntimeError('post-update inference produced no tokens')
    sample = result.sequences[0]
    if sample.logprobs is None or len(sample.logprobs) != len(sample.tokens) or any(not math.isfinite(p) for p in sample.logprobs):
        raise RuntimeError('post-update inference returned invalid logprobs')
    record('replay_complete', buckets=[c['bucket'] for c in fixture['cases']], backward_passes=2*len(fixture['cases']),
           optimizer_updates=1, post_update_tokens=len(sample.tokens), sampling_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--learning-rate', required=True, type=float)
    parser.add_argument('--fixture', default='gemma_attention_replay.json')
    run(parser.parse_args())
