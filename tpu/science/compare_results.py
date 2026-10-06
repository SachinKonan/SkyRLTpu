"""Summarize complete, matched model pilots without dropping invalid samples."""
import argparse
import hashlib
import json
from pathlib import Path


MATCHED_SETTINGS = (
    'prompt_sha256', 'context_window', 'phase1_prompt_plus_thinking',
    'temperature', 'top_p', 'top_k', 'sampling_seed', 'samples',
)


def compare(folders):
    settings = {}
    rows = []
    for folder in folders:
        experiment = json.loads((folder / 'experiment.json').read_text())
        for task in ('portfolio', 'routing'):
            root = folder / task
            generation = json.loads((root / 'generation-summary.json').read_text())
            current = {key: generation[key] for key in MATCHED_SETTINGS}
            if task in settings and current != settings[task]:
                raise ValueError(f'{folder.name}/{task}: generation settings or prompt differ')
            settings[task] = current
            candidates = generation['candidates']
            if len(candidates) != generation['samples']:
                raise ValueError(f'{root}: incomplete generation records')
            verdicts = []
            for candidate in candidates:
                index = candidate['index']
                result = json.loads((root / f'verdict-{index:03d}.json').read_text())
                if result['candidate_index'] != index or result['generation'] != candidate:
                    raise ValueError(f'{root}: mismatched candidate {index}')
                if candidate['has_code']:
                    source = (root / f'candidate-{index:03d}.py').read_bytes()
                    if hashlib.sha256(source).hexdigest() != result['source_sha256']:
                        raise ValueError(f'{root}: source hash mismatch for {index}')
                if not candidate['thinking_budget']['enforced']:
                    raise ValueError(f'{root}: native budget not enforced for {index}')
                verdicts.append(result)
            valid = [v for v in verdicts if v['correctness'] == 1]
            best = max(valid, key=lambda v: v['reward']) if valid else None
            metric = 'sharpe' if task == 'portfolio' else 'swaps'
            rows.append(dict(
                model=generation.get('model', experiment['model']),
                model_preset=generation.get('model_preset'),
                task=task, samples=len(verdicts), valid=len(valid),
                mean_reward=sum(v['reward'] for v in verdicts) / len(verdicts),
                best_reward=max(v['reward'] for v in verdicts),
                scientific_metric=metric,
                best_scientific_value=best['metrics'][metric] if best else None,
                best_candidate=best['candidate_index'] if best else None,
                generation_seconds=generation['generation_seconds'],
                failures=[dict(index=v['candidate_index'], reason=v['msg'])
                          for v in verdicts if v['correctness'] != 1],
            ))
    return dict(settings=settings, results=rows,
                portfolio_causality_status='Legacy v1 replay exposed next-open holdings to an earlier close-time decision. Portfolio scores are diagnostic only, pending corrected replay; routing results are unaffected.',
                interpretation='Four-sample inference pilot; no RL training or robust model ranking.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.folders)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
