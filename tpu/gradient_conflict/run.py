"""Frozen Qwen LoRA gradient comparison, driven through the real Tinker API."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tinker
from tinker import types

from skyrl.backends.gradient_probe import compare, dot, fingerprint


class DurableCache:
    """Immutable inputs identify resumable scoring and gradient artifacts."""

    def __init__(self, source, prefix):
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        self.local = Path(source).parent / 'cache-v5' / digest
        self.local.mkdir(parents=True, exist_ok=True)
        self.remote = prefix.rstrip('/') + '/cache-v5/' + digest

    def fetch(self, name):
        target = self.local / name
        target.parent.mkdir(parents=True, exist_ok=True)
        output = subprocess.run(['gcloud', 'storage', 'cp', self.remote+'/'+name, str(target)],
                                capture_output=True, text=True)
        if output.returncode == 0:
            return target
        error = (output.stdout + output.stderr).lower()
        if any(s in error for s in ('matched no objects', 'no urls matched', 'not found', 'does not exist', '404')):
            return None
        raise RuntimeError(f'Cache retrieval failed: {output.stderr}')

    def put(self, name, source):
        subprocess.run(['gcloud', 'storage', 'cp', str(source), self.remote+'/'+name], check=True)

    def save_scores(self, name, scores):
        path = self.local / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt') as stream:
            json.dump(scores, stream, allow_nan=False)
        self.put(name, path)


def make_datum(row):
    tokens = row['tokens']
    n = len(tokens) - 1
    prefix = row['prompt_tokens'] - 1
    # Production remove_mask() leaves weights at the API default of one,
    # and uses zero prompt advantages. Its denominator is full sequence length.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            'target_tokens': types.TensorData(data=tokens[1:], dtype='int64', shape=[n]),
            'weights': types.TensorData(data=[1.0] * n, dtype='float32', shape=[n]),
            'advantages': types.TensorData(data=[0.0] * prefix + [row['advantage']] * (n-prefix),
                                           dtype='float32', shape=[n]),
            'logprobs': types.TensorData(data=[0.0] * n, dtype='float32', shape=[n]),
        },
    )


def add(parts):
    nonempty = [p for p in parts if p is not None]
    if not nonempty:
        return None
    return {key: sum(np.asarray(p[key], dtype=np.float32) for p in nonempty) for key in nonempty[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    # Accepted for compatibility with the existing four-host smoke worker.
    parser.add_argument('--rows', type=int)
    parser.add_argument('--replays', type=int)
    parser.add_argument('--sequence-length', type=int)
    args = parser.parse_args()
    with gzip.open(os.environ['TUNIX_GRADIENT_PROBE_INPUT'], 'rt') as stream:
        data = json.load(stream)
    prefix = os.environ['GRADIENT_PROBE_RESULT_PREFIX']
    cache = DurableCache(os.environ['TUNIX_GRADIENT_PROBE_INPUT'], prefix)
    # A fresh API has an empty checkpoint registry even after the tarball is
    # staged. Use the same explicit registration step as production resumes.
    repo = Path(__file__).resolve().parents[2]
    source_model, kind, checkpoint_id = data['checkpoint'].removeprefix('tinker://').split('/')
    if kind != 'weights':
        raise ValueError('Expected a training checkpoint')
    subprocess.run([
        sys.executable, str(repo/'tpu/reregister_states.py'),
        '--db', str(repo/'skyrl/tinker/tinker.db'),
        '--ckpt-root', os.environ['REMOTE_CHECKPOINTS'],
        '--base-model', args.base_model, '--entry', f'{source_model}:{checkpoint_id}',
    ], check=True)
    # Register before creating the SDK session/model, so registration does not
    # contend with their startup writes and heartbeat transactions.
    service = tinker.ServiceClient(base_url='http://127.0.0.1:8000', api_key='tml-local-probe')
    trainer = service.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    trainer.load_state_with_optimizer(data['checkpoint']).result()
    probe_dir = Path(os.environ['TUNIX_GRADIENT_PROBE_DIR']) / trainer.model_id
    fingerprints = set()
    result = {k: v for k, v in data.items() if k != 'groups'}
    result.update(groups=[], seed=20260906, lora_rank=args.rank, parameter_space='trainable LoRA',
                  loss_normalization='sum of per-sequence losses divided by full sequence token counts',
                  cache_prefix=cache.remote,
                  accelerator=os.environ.get('GRADIENT_PROBE_ACCELERATOR', 'tpu-v4-32'),
                  train_workers=os.environ.get('TRAIN_WORKERS', '0,1,2,3'),
                  mesh={'tp': int(os.environ.get('TRAIN_TP_SIZE', 8)),
                        'fsdp': int(os.environ.get('TRAIN_FSDP_SIZE', 2))})

    def publish():
        result['parameter_sha256'] = next(iter(fingerprints))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        subprocess.run(['gcloud','storage','cp',str(args.output),prefix+'/progress.json'], check=True)

    def drain(datums, label, *, use_cache=True):
        if not datums:
            return None
        key = 'gradients/' + hashlib.sha256(label.encode()).hexdigest()
        cached = cache.fetch(key+'.json') if use_cache else None
        if cached:
            arrays_path = cache.fetch(key+'.npz')
            if arrays_path is None:
                raise RuntimeError('Completed gradient cache marker has no arrays')
            stem = cached.with_suffix('')
            print(f'{label}: restored cached gradient for {len(datums)} rows', flush=True)
        else:
            print(f'{label}: backward {len(datums)} rows', flush=True)
            # Requests stay small, and all accumulate at the identical parameters.
            for start in range(0, len(datums), 2):
                trainer.forward_backward(datums[start:start+2], 'importance_sampling').result()
                print(f'{label}: backward {min(start+2,len(datums))}/{len(datums)}', flush=True)
            output = trainer.optim_step(types.AdamParams(learning_rate=0.0, weight_decay=0.0)).result()
            index = int(output.metrics['gradient_probe/index'])
            stem = probe_dir / f'{index:06d}'
        record = json.loads(Path(str(stem)+'.json').read_text())
        if record['count'] != len(datums):
            raise RuntimeError('Gradient accumulation count mismatch')
        fingerprints.add(record['parameter_sha256'])
        if len(fingerprints) != 1:
            raise RuntimeError('Frozen model parameters changed')
        with np.load(str(stem)+'.npz') as arrays:
            values = {k: arrays[k] for k in arrays.files}
        if fingerprint(values) != record['gradient_sha256']:
            raise RuntimeError('Gradient artifact checksum mismatch')
        if use_cache and not cached:
            cache.put(key+'.npz', str(stem)+'.npz')
            # JSON is a completion marker, uploaded only after the arrays.
            cache.put(key+'.json', str(stem)+'.json')
        Path(str(stem)+'.npz').unlink()
        return values

    for group_idx, group in enumerate(data['groups']):
        started = time.monotonic()
        rows = group['rows']
        datums = [make_datum(row) for row in rows]
        print(f"GROUP {group['id']} forward-scoring {len(rows)} rows", flush=True)
        # Stop-gradient reference at this checkpoint: the IS derivative is the
        # signed score-function gradient. No old-policy correction is claimed.
        for start in range(0, len(datums), 2):
            subset = datums[start:start+2]
            score_key = f"scores/{group['id']}/{start:03d}.json.gz"
            cached_scores = cache.fetch(score_key)
            if cached_scores:
                with gzip.open(cached_scores, 'rt') as stream:
                    logprobs = json.load(stream)
            else:
                output = trainer.forward(subset, 'cross_entropy').result()
                logprobs = [out['logprobs'].data for out in output.loss_fn_outputs]
            if len(logprobs) != len(subset):
                raise RuntimeError('Missing forward rows')
            for datum, lp in zip(subset, logprobs):
                if len(lp) != len(datum.loss_fn_inputs['target_tokens'].data) or not np.isfinite(lp).all():
                    raise RuntimeError('Invalid reference logprobs')
                datum.loss_fn_inputs['logprobs'] = types.TensorData(data=lp, dtype='float32', shape=[len(lp)])
            if not cached_scores:
                cache.save_scores(score_key, logprobs)
            print(f"{group['id']}: scored {min(start+2,len(rows))}/{len(rows)}", flush=True)
            if group_idx == 0 and start == 0:
                # Always verify the freshly restored live parameters, including
                # on retries that otherwise consume cached gradients.
                drain(subset, 'Preflight frozen export', use_cache=False)
                print('Preflight frozen export passed', flush=True)
        valid = np.array([r['valid'] for r in rows])
        nv = int(valid.sum())
        random_left = np.zeros(len(rows), dtype=bool)
        rng = np.random.default_rng(20260906 + group_idx)
        random_left[rng.permutation(len(rows))[:nv]] = True
        # Four intersections yield validity and random-split gradients in one
        # backward pass per sample. No per-sample gradient needs to be stored.
        buckets = {}
        for v in (False, True):
            for r in (False, True):
                selected = [d for i,d in enumerate(datums) if valid[i] == v and random_left[i] == r]
                buckets[v,r] = drain(selected, f"{group['id']} valid={v} random={r}")
        gv = add([buckets[True,False], buckets[True,True]])
        gi = add([buckets[False,False], buckets[False,True]])
        ra = add([buckets[False,True], buckets[True,True]])
        rb = add([buckets[False,False], buckets[True,False]])
        stats = {k:v for k,v in group.items() if k != 'rows'}
        stats['valid_vs_invalid'] = compare(gv, gi, nv, len(rows)-nv)
        stats['random_split'] = compare(ra, rb, nv, len(rows)-nv)
        stats['valid_completion_tokens'] = sum(r['completion_tokens'] for r in rows if r['valid'])
        stats['invalid_completion_tokens'] = sum(r['completion_tokens'] for r in rows if not r['valid'])
        stats['per_leaf'] = {k: compare({k:gv[k]}, {k:gi[k]}, nv, len(rows)-nv) for k in gv}
        result['groups'].append(stats)
        # Independent full-group backward verifies the partition arithmetic.
        if group_idx == 0:
            stats['reconstruction_status'] = 'pending'
            publish()
            full = drain(datums, group['id']+' reconstruction')
            delta = {k: full[k] - gv[k] - gi[k] for k in full}
            error = np.sqrt(dot(delta, delta) / max(dot(full, full), 1e-30))
            stats['reconstruction_relative_error'] = float(error)
            stats['reconstruction_status'] = 'passed' if error <= 0.02 else 'failed'
            publish()
            if error > 0.02:
                raise RuntimeError(f'Gradient reconstruction failed: {error}')
        stats['seconds'] = time.monotonic() - started
        publish()
        print(json.dumps({"group":group['id'], **stats['valid_vs_invalid']}), flush=True)
    result['complete'] = True
    result['groups_completed'] = len(result['groups'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('GRADIENT PROBE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
