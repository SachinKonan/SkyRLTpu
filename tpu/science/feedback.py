"""Compact diagnostics for both saved candidates and invalid-only repair."""
import json
import re


def feedback_limit(task):
    return 10000 if task == 'placement' else 3000


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
        'benchmark_suite', 'required_case_count', 'leaderboard_verified')}
    message = result['msg'][:1600]
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
            if 'message' in case:
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
