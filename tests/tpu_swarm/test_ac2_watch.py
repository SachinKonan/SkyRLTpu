import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / 'tpu/science/results/campaign-next-20260921/watch_ac2.py'
spec = importlib.util.spec_from_file_location('ac2_watch', PATH)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def receipts():
    return [dict(run_id=run, state='submitted', job_id=100 + i)
            for i, run in enumerate(sorted(watch.RUNS))]


def test_partial_submission_is_not_completion():
    assert watch.receipt_state([]) == 'empty'
    assert watch.receipt_state(receipts()[:1]) == 'partial'
    assert watch.receipt_state(receipts()) == 'submitted'


@pytest.mark.parametrize('change', [dict(state='submitting'), dict(job_id=None), dict(job_id=True)])
def test_uncertain_submission_requires_reconciliation(change):
    rows = receipts()
    rows[0].update(change)
    with pytest.raises(ValueError, match='uncertain submission'):
        watch.receipt_state(rows)


def test_duplicate_run_and_job_ids_are_rejected():
    rows = receipts()
    with pytest.raises(ValueError, match='duplicate'):
        watch.receipt_state(rows + rows[:1])
    rows[1]['job_id'] = rows[0]['job_id']
    with pytest.raises(ValueError, match='same job ID'):
        watch.receipt_state(rows)


def test_unrelated_receipts_are_not_accepted():
    rows = receipts()
    rows[0]['run_id'] = 'unrelated-run'
    with pytest.raises(ValueError, match='unexpected'):
        watch.receipt_state(rows)


def setup_watch(tmp_path, monkeypatch):
    for name, child in [('ROOT', 'root'), ('HERE', 'scripts'), ('OUT', 'out'), ('DEST', 'dest')]:
        path = tmp_path / child
        path.mkdir()
        monkeypatch.setattr(watch, name, path)
    guard = dict(head='reviewed', diff_sha256='clean')
    watch.write(watch.OUT / 'source-guard.json', guard)
    monkeypatch.setattr(watch, 'source_guard', lambda: guard)
    monkeypatch.setattr(sys, 'argv', ['watch_ac2', '--once'])
    calls = []
    monkeypatch.setattr(watch.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    return calls


def test_live_unfinished_gate_cannot_build_or_submit(tmp_path, monkeypatch):
    calls = setup_watch(tmp_path, monkeypatch)
    watch.write(watch.HERE / 'ac2-completion.json', dict(ready=False))
    watch.main()
    assert len(calls) == 1 and calls[0][-1].endswith('check_completion.py')
    assert json.loads((watch.OUT / 'watch-state.json').read_text())['phase'] == 'waiting'


def test_pending_intent_stops_without_relaunch(tmp_path, monkeypatch):
    calls = setup_watch(tmp_path, monkeypatch)
    row = receipts()[0]
    row['state'] = 'submitting'
    watch.write(watch.DEST / 'submissions.json', [row])
    with pytest.raises(ValueError, match='uncertain submission'):
        watch.main()
    assert calls == []
    assert json.loads((watch.OUT / 'watch-state.json').read_text())['phase'] == 'needs_review'


def test_source_change_stops_before_preparation(tmp_path, monkeypatch):
    calls = setup_watch(tmp_path, monkeypatch)
    monkeypatch.setattr(watch, 'source_guard', lambda: dict(head='unreviewed'))
    with pytest.raises(RuntimeError, match='source changed'):
        watch.main()
    assert calls == []


def test_receipts_need_matching_live_queue_jobs():
    rows = receipts()
    queue = [dict(job_id=r['job_id'], run_id=r['run_id'], status='PENDING') for r in rows]
    watch.verify_queue(rows, queue)
    with pytest.raises(ValueError, match='live queue'):
        watch.verify_queue(rows, queue[:1])
    queue[0]['run_id'] = 'unrelated'
    with pytest.raises(ValueError, match='live queue'):
        watch.verify_queue(rows, queue)
    queue[0]['run_id'] = rows[0]['run_id']
    queue[0]['status'] = 'FAILED'
    with pytest.raises(ValueError, match='failed or was cancelled'):
        watch.verify_queue(rows, queue)


def test_guard_ignores_observations_but_detects_recipe_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'ROOT', tmp_path)
    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, check=True,
                              capture_output=True)
    git('init')
    paths = [
        'tpu/science/results/campaign-next-20260921/ac2-completion.json',
        'tpu/science/results/reallocation-10step-20260921/circuit-queue-state.json',
        'tpu/science/results/reallocation-10step-20260921/placement-v5p64-gemma-10step-20260921-launch.json',
        'tpu/swarm/ray_train/profiles/recipe.json',
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-m', 'initial')
    baseline = watch.source_guard()
    for name in paths[:-1]:
        (tmp_path / name).write_text('{"phase": "updated"}\n')
    assert watch.source_guard() == baseline
    (tmp_path / paths[-1]).write_text('{"epochs": 20}\n')
    assert watch.source_guard() != baseline
