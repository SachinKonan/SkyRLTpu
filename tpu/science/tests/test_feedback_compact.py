import json

from tpu.science.feedback import compact


def _feedback(valid=True):
    cases = [dict(case='ibm01', valid=True, candidate_limit_seconds=310, proxy_cost=0.755314,
                  wirelength_cost=0.0719638, density_cost=0.485071, congestion_cost=0.881629,
                  candidate_wall_seconds=299.094, grading_seconds=6.39064, overlap_count=0.0),
             dict(case='ibm02', valid=valid, candidate_limit_seconds=310, proxy_cost=0.916643,
                  wirelength_cost=0.0782528, density_cost=0.518113, congestion_cost=1.15867,
                  candidate_wall_seconds=299.113, grading_seconds=8.75946, overlap_count=0.0)]
    if not valid:
        cases[1]['message'] = 'hard macros 3 and 7 overlap'
    return json.dumps(dict(reward=0.5115 if valid else 0.0, message='Valid' if valid else 'Invalid placement',
                           metrics=dict(mean_proxy_cost=0.954891, benchmark_suite='ibm17-proxy-v1', case_count=2,
                                        cases=cases)))


def test_placement_feedback_becomes_one_header_and_one_row_per_case():
    text = compact('placement', _feedback())
    lines = text.splitlines()
    assert lines[0].startswith('reward=0.5115 message=Valid mean_proxy_cost=0.9549')
    assert lines[1] == 'case,proxy_cost,wirelength_cost,density_cost,congestion_cost,candidate_wall_seconds'
    assert lines[2] == 'ibm01,0.7553,0.07196,0.4851,0.8816,299.1'
    assert len(lines) == 4 and 'grading_seconds' not in text and 'candidate_limit_seconds' not in text


def test_invalid_cases_keep_their_error():
    text = compact('placement', _feedback(valid=False))
    assert text.splitlines()[-2:] == ['invalid cases:', 'ibm02: hard macros 3 and 7 overlap']


def test_other_tasks_and_non_json_pass_through():
    assert compact('routing', '{"a":1}') == '{"a":1}'
    assert compact('placement', 'TimeoutError: boom') == 'TimeoutError: boom'
