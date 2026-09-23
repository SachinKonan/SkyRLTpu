"""Compact diagnostics for both saved candidates and invalid-only repair."""
import json
import re


def feedback_limit(task):
    if task == 'routing':
        return 12000
    return 10000 if task == 'placement' else 3000


def routing_cases(metrics):
    """Compact, lossless counts from trusted verification, including old journals.

    Column names are supplied once to avoid repeating four keys for 72 cases.
    Baselines for older verdicts come from the pinned public suite, never from
    candidate output. Missing measurements remain null, not invented zeros.
    """
    from pathlib import Path
    from .routing_suite import manifest
    specs = {row['id']: row for row in manifest()['cases']}
    cases, topologies = [], {}
    for row in metrics['cases']:
        spec = specs.get(row['case'], {})
        baseline_cnots = row.get('baseline_added_cnots', spec.get('original_cnot_added'))
        baseline = baseline_cnots / 3 if baseline_cnots is not None else None
        if baseline is not None and baseline.is_integer():
            baseline = int(baseline)
        swaps = row['swaps']
        delta = swaps - baseline if baseline is not None else None
        cases.append([row['case'], swaps, baseline, delta])
        topology = row.get('topology') or Path(spec.get('topology_path', 'unknown')).stem
        totals = topologies.setdefault(topology, dict(case_count=0, swaps=0, baseline_swaps=0))
        totals['case_count'] += 1
        totals['swaps'] += swaps
        if baseline is None:
            totals['baseline_swaps'] = None
        elif totals['baseline_swaps'] is not None:
            totals['baseline_swaps'] += baseline
    from .routing_resources import GEMINI_TARGETS
    for topology, totals in topologies.items():
        expected = {name for name, spec in specs.items() if Path(spec.get('topology_path', 'unknown')).stem == topology}
        observed = [row[0] for row in cases if row[0] in expected]
        totals['complete'] = len(observed) == len(expected) == 24 and len(set(observed)) == 24
        if topology in GEMINI_TARGETS:
            totals['gemini_target_swaps'] = GEMINI_TARGETS[topology]
            totals['gap_to_gemini'] = totals['swaps'] - GEMINI_TARGETS[topology] if totals['complete'] else None
        baseline = totals['baseline_swaps']
        totals['delta_swaps'] = totals['swaps'] - baseline if baseline is not None else None
    columns=['case', 'swaps', 'baseline_swaps', 'delta_swaps']
    if 'case_statuses' in metrics:
        measured={row[0]:row for row in cases}
        statuses={row['case']:row['status'] for row in metrics['case_statuses']}
        cases=[]
        for name,spec in specs.items():
            values=measured.get(name,[name,None,spec['original_cnot_added']/3,None])
            cases.append([name,statuses.get(name,'not_started'),*values[1:]])
        columns.insert(1,'status')
    return dict(case_columns=columns, cases=cases, topologies=topologies)


def case_error(row):
    """Keep the terminal exception, not the subprocess command wrapping it.

    Logs remain diagnostic data, never instructions or executable content.
    Older journals retain logs, so the same formatter can repair their feedback.
    """
    message = row['msg']
    metrics = row.get('metrics', {})
    log = metrics.get('grader.log') or metrics.get('candidate.log', '')
    errors = re.findall(r'^((?:[\w.]+(?:Error|Exception)|TimeoutExpired):[^\n]*)',
                        log, flags=re.MULTILINE)
    if errors:
        message = errors[-1]
    return message[:360]


def observation(task, result):
    metrics = result['metrics']
    feedback = {k: v for k, v in metrics.items() if k in (
        'mean_proxy_cost', 'weighted_candidate_cnots', 'weighted_baseline_cnots',
        'swaps', 'added_cnots', 'improvement', 'case_count', 'total_seconds', 'routing_suite',
        'benchmark_suite', 'required_case_count', 'leaderboard_verified',
        'completed_cases', 'suite_complete')}
    message = result['msg'][:1600]
    if task == 'routing' and 'cases' in metrics:
        feedback.update(routing_cases(metrics))
        if 'case_statuses' in metrics:
            feedback['status_counts']={}
            for row in metrics['case_statuses']:
                status=row['status']
                feedback['status_counts'][status]=feedback['status_counts'].get(status,0)+1
            feedback['case_errors']={name:detail[-240:] for name,detail in metrics.get('case_errors',{}).items()}
    if task == 'placement' and 'cases' in metrics:
        from .challenge_contract import CASES, CANDIDATE_LIMIT_SECONDS
        by_case = {row['metrics'].get('case'): row for row in metrics['cases']}
        cases = []
        for case in CASES:
            if case not in by_case:
                continue
            row = by_case[case]
            item = {'case': case, 'valid': row['correctness'] == 1,
                    'candidate_limit_seconds': row['metrics'].get(
                        'candidate_limit_seconds', CANDIDATE_LIMIT_SECONDS)}
            for key in ('proxy_cost', 'wirelength_cost', 'density_cost', 'congestion_cost',
                        'candidate_wall_seconds', 'grading_seconds', 'overlap_count'):
                if key in row['metrics']:
                    item[key] = float(format(row['metrics'][key], '.6g'))
            if not item['valid']:
                item['message'] = case_error(row)
            cases.append(item)
        feedback['cases'] = cases
        message = 'Valid' if result['correctness'] == 1 else 'Invalid placement; see per-case errors.'
    # Bound fields before serialization, so feedback is always complete JSON.
    value = dict(reward=result['reward'], message=message, metrics=feedback)
    encoded = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
    if len(encoded) > feedback_limit(task):
        value['message'] = message[:600]
        for case in feedback.get('cases', []):
            if isinstance(case, dict) and 'message' in case:
                case['message'] = case['message'][:160]
        encoded = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
    if len(encoded) > feedback_limit(task):
        raise ValueError('science feedback exceeded its diagnostic budget')
    return encoded


def diagnostic_message(task, result):
    if task == 'placement' and result['correctness'] != 1 and 'cases' in result['metrics']:
        return '; '.join(f"{row['metrics'].get('case')}: {case_error(row)}"
                         for row in result['metrics']['cases'] if row['correctness'] != 1)
    return result['msg']
