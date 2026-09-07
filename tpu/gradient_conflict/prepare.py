"""Select reproducible mixed-validity groups from archived Qwen trajectories.

Archives do not contain sampled token IDs/logprobs. This explicitly builds a
text-reconstructed, frozen-checkpoint score-function diagnostic, not a replay
of historical importance sampling. Original group baselines are retained.
"""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def prepare(archive_dir, tokenizer_path, max_length=22528, num_groups=6):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    candidates = []
    sources = []
    for path in sorted(Path(archive_dir).glob('qwen_step_*.jsonl.gz')):
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        groups = collections.defaultdict(list)
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                groups[row['seed_idx']].append(row)
        for seed, rows in sorted(groups.items()):
            rewards = np.array([r['reward'] for r in rows], dtype=np.float64)
            if not np.isfinite(rewards).all():
                raise ValueError('Non-finite archived reward')
            baseline = float(rewards.mean())
            if np.ptp(rewards) == 0:
                continue
            retained, excluded = [], []
            for index, row in enumerate(rows):
                if row['correctness'] not in (0, 1):
                    raise ValueError('Expected binary evaluator correctness')
                # Match the Qwen3Renderer's separately tokenized chunks.
                prompt = (encode('<|im_start|>user\n') + encode(row['prompt'] + '<|im_end|>')
                          + encode('\n<|im_start|>assistant\n<think>\n'))
                response = encode(row['response_raw'])
                tokens = prompt + response
                reason = 'empty_completion' if not response else 'too_long' if len(tokens) > max_length else None
                if reason:
                    excluded.append({"index": index, "valid": bool(row['correctness']),
                                     "reason": reason, "tokens": len(tokens)})
                    continue
                retained.append({
                    "index": index, "valid": bool(row['correctness']), "reward": row['reward'],
                    "advantage": float(row['reward'] - baseline),
                    "tokens": tokens, "prompt_tokens": len(prompt),
                    "completion_tokens": len(response),
                })
            nvalid = sum(r['valid'] for r in retained)
            if nvalid == 0 or nvalid == len(retained):
                continue
            candidates.append({
                "id": f"step{rows[0]['step']:06d}-seed{seed:02d}",
                "step": rows[0]['step'], "seed_idx": seed, "source": path.name,
                "original_count": len(rows), "original_valid_count": sum(bool(r['correctness']) for r in rows),
                "baseline": baseline, "excluded": excluded, "rows": retained,
                "valid_fraction": nvalid / len(retained),
            })
    candidates.sort(key=lambda g: (g['valid_fraction'], g['id']))
    if len(candidates) < num_groups:
        raise ValueError(f'Only {len(candidates)} eligible mixed groups; need {num_groups}')
    indices = np.linspace(0, len(candidates) - 1, num_groups).round().astype(int)
    return {
        "schema": 1, "model": 'Qwen/Qwen3.5-27B',
        "checkpoint": 'tinker://model_4ee1d2d2/weights/000003',
        "objective": 'frozen_checkpoint_advantage_weighted_loglikelihood',
        "advantage_estimator": 'mean_baseline', "sources": sources,
        "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        "limitations": [
            'Text retokenization; original sampled IDs and sampling logprobs unavailable.',
            'Original termination tokens and phase-two action masks unavailable; no EOS is fabricated.',
            'Frozen-checkpoint logprobs replace historical behavior logprobs (ratio one at measurement).',
            'Six selected 32-rollout groups, not six full optimizer batches; length exclusions are reported.',
        ],
        "max_length": max_length, "candidate_groups": len(candidates),
        "groups": [candidates[i] for i in indices],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-dir', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = prepare(args.archive_dir, args.tokenizer)
    with gzip.open(args.output, 'wt') as stream:
        json.dump(data, stream)
    for group in data['groups']:
        print(group['id'], 'rows', len(group['rows']), 'valid', group['valid_fraction'],
              'excluded', len(group['excluded']), flush=True)
