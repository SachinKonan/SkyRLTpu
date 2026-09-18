"""Import Q20 seeds by rescoring archived trusted full-suite bootstrap journals."""
import argparse
import hashlib
from pathlib import Path

from .bootstrap import commit_rows, identity, read, save
from .feedback import observation
from .routing_suite import rescore_verified
from .seed_pool import verify_pool


def rebuild(clients, destination):
    from ttt_discover import State
    from ttt_discover.tinker_utils.sampler import PUCTSampler
    from .training_env import RoutingTrainingEnv

    destination = Path(destination)
    if destination.exists():
        raise ValueError('refusing to overwrite an imported seed pool')
    roots, rows, sources, ids = {}, [], [], set()
    excluded = 0
    for client in map(Path, clients):
        folder = client / 'bootstrap'
        for value in read(folder / 'roots.json'):
            state = State.from_dict(value)
            if state.id in roots or state.code or state.value != 0:
                raise ValueError('duplicate or noninitial root')
            roots[state.id] = state
        for path in sorted(folder.glob('layer-*/group-*/grade-*.json')):
            row = read(path)
            if row['id'] in ids or row['root_id'] not in roots:
                raise ValueError('duplicate candidate or missing root')
            ids.add(row['id'])
            if row['correctness'] != 1:
                excluded += 1
                continue
            result = rescore_verified(row)
            sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                candidate_id=row['id'], original_reward=row['reward']))
            row.update(reward=result['reward'], metrics=result['metrics'],
                       feedback=observation('routing', result), message=result['msg'])
            rows.append(row)
    if len(roots) != 16 or not rows:
        raise ValueError('expected sixteen roots and verified seed programs')
    path = destination / 'puct_sampler_step_000000.json'
    save(path, dict(step=0, states=[s.to_dict() for s in roots.values()],
                   initial_states=[s.to_dict() for s in roots.values()], puct_n={}, puct_m={}, puct_T=0))
    sampler = PUCTSampler(str(destination / 'puct_sampler.json'), env_type=RoutingTrainingEnv,
                          problem_type='routing', batch_size=16, resume_step=0)
    commit_rows(sampler, roots, rows)
    pool = read(path)
    digest = identity(pool)
    verify_pool(path, digest)
    retained = [s for s in pool['states'] if s.get('code')]
    best = max(rows, key=lambda r: r['reward'])
    report = dict(pool_sha256=digest, routing_suite='q20', case_count=24, valid=len(rows),
                  distinct_valid=len({r['code'] for r in rows}), retained=len(retained), maximum=32,
                  best_reward=best['reward'], best_q20_swaps=best['metrics']['swaps'],
                  optimizer_steps=0, model_initialization='pretrained',
                  excluded_full_suite_failures=excluded,
                  admission='Top two per original root, global code deduplication; Q20 rewards only',
                  failure_policy='Full-suite failures not assumed invalid on Q20; excluded without imported visits',
                  sources=sources)
    save(destination / 'seed-import.json', report)
    save(destination / 'rescored-candidates.json', rows)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client', action='append', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    report = rebuild(args.client, args.destination)
    print({k: v for k, v in report.items() if k != 'sources'})
