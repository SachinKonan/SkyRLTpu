"""Wait for the original AC2 round, then prepare and submit its shared winner.

Run with --arm once after review to pin HEAD and the tracked working diff.
The service never cancels jobs. Uncertain submission intent requires manual
reconciliation; it is never automatically retried or declared successful.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
OUT = ROOT / '.science/ac2-shared-best-20260921'
DEST = HERE.parent / 'ac2-shared-best-20260921'
MODELS = {'qwen', 'gemma', 'muse'}
RUNS = {f'ac2-shared-best-{m}-10step-20260921' for m in MODELS}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def source_guard():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)
    # This tracked observation file changes on every completion poll.
    diff = git('diff', '--binary', 'HEAD', '--', '.',
               ':(exclude)tpu/science/results/campaign-next-20260921/ac2-completion.json')
    return dict(head=git('rev-parse', 'HEAD').decode().strip(),
                diff_sha256=hashlib.sha256(diff).hexdigest())


def receipt_state(rows):
    names = [r.get('run_id') for r in rows]
    if len(set(names)) != len(names) or not set(names) <= RUNS:
        raise ValueError('unexpected or duplicate handoff receipts')
    if any(r.get('state') != 'submitted' or type(r.get('job_id')) is not int
           or r['job_id'] <= 0 for r in rows):
        raise ValueError('uncertain submission; reconcile receipts before continuing')
    ids = [r['job_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('multiple handoff runs have the same job ID')
    return 'submitted' if set(names) == RUNS else 'partial' if rows else 'empty'


def verify_queue(receipts, queue):
    for receipt in receipts:
        matches = [r for r in queue if r['job_id'] == receipt['job_id']]
        if len(matches) != 1 or matches[0]['run_id'] != receipt['run_id']:
            raise ValueError('submission receipt does not match the live queue')
        if matches[0]['status'] not in {'PENDING', 'SUBMITTED', 'STARTING', 'RUNNING', 'RECOVERING', 'SUCCEEDED'}:
            raise ValueError('handoff job failed or was cancelled; inspect before continuing')


def validate_prepared():
    # Called on a CPU allocation, including the numerical winner verification.
    from tpu.science.ac2_handoff import best_completed_candidate, recipient_profile
    pool, provenance = best_completed_candidate(json.loads((OUT / 'snapshots.json').read_text()))
    assert json.loads((DEST / 'provenance.json').read_text()) == provenance
    rows = json.loads((DEST / 'prepared.json').read_text())['jobs']
    assert len(rows) == 3 and {r['run_id'] for r in rows} == RUNS
    originals = json.loads((HERE.parent / 'reallocation-10step-20260921/prepared.json').read_text())['jobs']
    seed = json.dumps(pool, indent=2).encode()
    for row in rows:
        model = row['model']
        assert row['run_id'] == f'ac2-shared-best-{model}-10step-20260921'
        source = next(r for r in originals if r['kind'] == 'ac2' and r['model'] == model)
        zone = 'us-east5-b' if model == 'qwen' else 'us-central1-b'
        bucket = 'gs://sk7524-tinker-tpu-us-central1'
        expected = recipient_profile(json.loads((ROOT / source['profile']).read_text()),
                                     row['run_id'], zone, bucket)
        assert json.loads((ROOT / row['profile']).read_text()) == expected
        assert row['epochs'] == 10 and row['priority'] == 120
        assert row['pool'] == 'tpuswarm-v6e32-' + ('east5b-qwen35' if model == 'qwen' else 'central1b')
        assert row['adapter'] == row['optimizer'] == 'fresh'
        assert (ROOT / row['seed_file']).read_bytes() == seed
        assert row['seed_pool_sha256'] == provenance['seed_pool_sha256']
        for field, digest in [('archive', 'archive_sha256'), ('task', 'task_sha256'),
                              ('seed_file', 'seed_file_sha256')]:
            assert hashlib.sha256((ROOT / row[field]).read_bytes()).hexdigest() == row[digest]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        validate_prepared()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.arm:
        guard_path = OUT / 'source-guard.json'
        if guard_path.exists():
            raise RuntimeError('already armed; inspect before replacing the source guard')
        write(guard_path, source_guard())
        return
    guard = json.loads((OUT / 'source-guard.json').read_text())
    python = str(ROOT / '.venv/bin/python')

    def run(*command):
        subprocess.run(command, cwd=ROOT, check=True)

    def cpu(*command):
        run('srun', '-p', 'cpu', '--cpus-per-task=2', '--mem=8G', '--time=00:30:00', *command)

    def confirm(receipts):
        spec = importlib.util.spec_from_file_location('ac2_gate', HERE / 'check_completion.py')
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        queue = gate.o.queue()
        verify_queue(receipts, queue)
        write(OUT / 'watch-state.json', dict(time=time.time(), phase='submitted', receipts=receipts))
        write(DEST / 'queue-after.json', queue)

    try:
        while True:
            if source_guard() != guard:
                raise RuntimeError('reviewed source changed; revalidate before rearming')
            receipts_path = DEST / 'submissions.json'
            receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else []
            state = receipt_state(receipts)
            if state == 'submitted':
                confirm(receipts)
                return
            # Poll failures are observation failures, never a reason to restart a job.
            try:
                run(python, str(HERE / 'check_completion.py'))
            except subprocess.CalledProcessError as exc:
                write(OUT / 'watch-state.json', dict(time=time.time(), phase='poll_error', error=str(exc)))
                if args.once:
                    raise
                time.sleep(60)
                continue
            if not json.loads((HERE / 'ac2-completion.json').read_text())['ready']:
                write(OUT / 'watch-state.json', dict(time=time.time(), phase='waiting'))
                if args.once:
                    return
                time.sleep(60)
                continue
            run(python, str(HERE / 'prepare_ac2_handoff.py'), 'fetch')
            if not (DEST / 'prepared.json').exists():
                if receipts:
                    raise RuntimeError('receipts exist without a prepared manifest')
                cpu(python, '-m', 'tpu.science.results.campaign-next-20260921.prepare_ac2_handoff', 'build')
            cpu(python, '-m', 'tpu.science.results.campaign-next-20260921.watch_ac2', '--validate-only')
            if source_guard() != guard:
                raise RuntimeError('source changed during preparation; refusing submission')
            run(python, str(HERE / 'submit.py'), '--ac2-handoff')
            receipts = json.loads(receipts_path.read_text())
            if receipt_state(receipts) != 'submitted':
                raise RuntimeError('not all three submissions have confirmed receipts')
            confirm(receipts)
            return
    except Exception as exc:
        write(OUT / 'watch-state.json', dict(time=time.time(), phase='needs_review', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
