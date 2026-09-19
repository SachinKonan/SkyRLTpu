"""Aggregate only complete, independently verified replay results."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).parent
manifest = json.loads((root / 'manifest.json').read_text())
specs = json.loads((root.parents[1] / 'manifests/routing-v1.json').read_text())['cases']
expected = {r['id']: r for r in specs}
summary = []
for record in manifest:
    model = record['model']
    result = json.loads((root / f'{model}-replay.json').read_text())
    metrics = result['metrics']
    assert result['correctness'] == 1 and metrics['case_count'] == 72
    assert metrics['routing_suite'] == 'full'
    source_hash = hashlib.sha256((root / f'{model}.py').read_bytes()).hexdigest()
    assert source_hash == record['source_sha256'] == metrics['source_sha256']
    assert metrics['seed'] == 42 and metrics['layout_trials'] == metrics['routing_trials'] == 20
    cases = metrics['cases']
    assert len(cases) == len({r['case'] for r in cases}) == 72
    assert {r['case'] for r in cases} == set(expected)
    totals = dict(Q20=0, Willow=0, Heron=0)
    counts = dict(Q20=0, Willow=0, Heron=0)
    weighted = 0.
    for row in cases:
        case = row['case']
        assert type(row['swaps']) is int and row['swaps'] >= 0
        assert row['added_cnots'] == 3 * row['swaps']
        name = 'Q20' if case.endswith('_q20') else 'Willow' if case.endswith('_willow') else 'Heron'
        totals[name] += row['swaps']; counts[name] += 1
        weighted += expected[case]['weight'] * row['added_cnots']
    assert set(counts.values()) == {24}
    assert sum(totals.values()) == metrics['swaps']
    assert math.isclose(weighted, metrics['weighted_candidate_cnots'], abs_tol=1e-8)
    summary.append(dict(model=model,swaps=totals,total_swaps=sum(totals.values()),
        reward=result['reward'],recorded_reward=record['recorded_reward'],
        recorded_total_swaps=record['recorded']['metrics']['swaps'],
        valid_cases=72,total_seconds=metrics['total_seconds'],source_sha256=source_hash))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for r in summary: print(r['model'],r['swaps'],r['total_swaps'],r['reward'])
