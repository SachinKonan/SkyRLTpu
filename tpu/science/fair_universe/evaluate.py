"""Trusted trial coordinator; candidate sees only explicitly mounted inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import numpy as np
from .contract import VENDOR, NOISE_SIGMA, bundle_identity, checked_scores, check_independence, digest
from .data import DataView


class CandidateFailure(RuntimeError):
    pass


def artifact_hashes(folder):
    result = {}
    for p in sorted(Path(folder).rglob('*')):
        if p.is_symlink():
            raise CandidateFailure('Symlinks are not allowed in detector artifacts')
        if p.is_file():
            result[str(p.relative_to(folder))] = digest(p)
    return result


def command(candidate, run, dataset, python, phase, chip):
    from tpu.science.isolation import command as isolated, python_mounts
    from tpu.science.placement_slots import device_paths, tpu_environment
    package = Path(__file__).parent
    readonly = python_mounts(python) + [(package, '/code/tpu/science/fair_universe'),
        (VENDOR, '/vendor'), (candidate, '/candidate.py'), (run/'config.json', '/config.json'),
        # libtpu/tcmalloc discovers CPU and accelerator topology through sysfs.
        # Match the existing TPU placement sandbox's read-only runtime mounts.
        ('/sys', '/sys'), ('/etc/hosts', '/etc/hosts')]
    # The lab venv may inherit a pinned, read-only preinstalled serving runtime.
    site_paths = json.loads(subprocess.check_output([str(python), '-c',
        'import sys,json;print(json.dumps([p for p in sys.path if p.endswith("/site-packages")]))'], text=True))
    inherited_prefixes = {str(Path(p).parents[2]) for p in site_paths}
    readonly += [(p, p) for p in inherited_prefixes if not any(Path(p).is_relative_to(Path(src)) for src, _ in readonly)]
    # uv also installs an unversioned interpreter-prefix symlink. Preserve its
    # lexical target inside the namespace, not just sys.base_prefix's realpath.
    executable = Path(python).absolute()
    while executable.is_symlink():
        target = Path(os.readlink(executable))
        executable = target if target.is_absolute() else executable.parent/target
        if target.is_absolute():
            prefix = str(target.parent.parent)
            if prefix not in {dst for _, dst in readonly}:
                readonly.append((prefix, prefix))
    writable = [(run/'cache', '/cache')]
    if phase == 'fit':
        readonly += [(dataset/'fit', '/fit')]
        writable += [(run/'artifacts', '/artifacts')]
    else:
        readonly += [(run/'artifacts', '/artifacts'), (run/'input', '/input'),
                     (dataset/'fit/val/mask.npy', '/mask.npy')]
    env = dict(PYTHONPATH='/code:/vendor',
               JAX_DEFAULT_MATMUL_PRECISION='highest',
               JAX_COMPILATION_CACHE_DIR='/cache/jax', XDG_CACHE_HOME='/cache',
               **tpu_environment(chip, 'tpu-v5p-32'))
    # The namespace exposes one physical VFIO device. libtpu enumerates it
    # as device zero, regardless of the host's assigned physical chip number.
    env.update(TPU_VISIBLE_CHIPS='0', TPU_CHIPS_PER_HOST_BOUNDS='1,1,1')
    cmd = isolated([python, '-m', 'tpu.science.fair_universe.child', phase],
                   readonly=readonly, writable=writable, env=env)
    # Device ordinals may be remapped by libtpu; verify on the actual v5p pilot.
    index = cmd.index('--dev') + 2
    devices = device_paths(chip, 'tpu-v5p-32')
    cmd[index:index] = [v for d in devices for v in ('--dev-bind', str(d), str(d))] + ['--tmpfs', '/dev/shm']
    return cmd


def kill(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=20)


class Predictor:
    def __init__(self, cmd, run, timeout):
        self.run = run
        self.deadline = time.monotonic() + timeout
        self.log = (run/'predict.log').open('wb')
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        self.buffer = b''
        try:
            if self.receive() != {'ready': True}:
                raise CandidateFailure('predictor initialization protocol failed')
        except BaseException:
            self.close(); raise

    def receive(self):
        while b'\n' not in self.buffer:
            if len(self.buffer) > 2**20:
                raise CandidateFailure('oversized prediction response')
            if not self.selector.select(max(0, self.deadline-time.monotonic())):
                raise CandidateFailure('prediction budget exceeded')
            block = os.read(self.proc.stdout.fileno(), 65536)
            if not block:
                raise CandidateFailure('predictor exited: ' + (self.run/'predict.log').read_text(errors='replace')[-3000:])
            self.buffer += block
        line, self.buffer = self.buffer.split(b'\n', 1)
        try:
            return json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise CandidateFailure('invalid prediction JSON') from exc

    def __call__(self, maps):
        np.save(self.run/'input/maps.npy', maps)
        self.proc.stdin.write(b'{"predict":true}\n'); self.proc.stdin.flush()
        try:
            return checked_scores(self.receive()['scores'], len(maps))
        except (ValueError, TypeError, KeyError) as exc:
            raise CandidateFailure('invalid prediction scores') from exc

    def close(self):
        kill(self.proc); self.selector.close(); self.log.close()


def evaluate(candidate, dataset, run, python, chip=0, seed=0, steps=250,
             batch_size=16, max_ind=3232, train_timeout=3600, reward_timeout=1800):
    candidate, dataset, run = Path(candidate).resolve(), Path(dataset).resolve(), Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    for name in ('cache', 'artifacts', 'input'):
        (run/name).mkdir()
    config = json.loads((VENDOR/'configs/default.json').read_text())
    for key in list(config):
        if key.startswith('reward_'):
            config.pop(key)
    config.update(seed=seed, steps=steps, batch_size=batch_size)
    (run/'config.json').write_text(json.dumps(config))
    manifest = json.loads((dataset/'manifest.json').read_text())
    identity = bundle_identity(candidate, config, manifest)
    started = time.monotonic()
    result = dict(identity=identity, candidate_sha256=digest(candidate), seed=seed,
                  reward_is_proxy_not_official=True, production_reward_selected=False)
    predictor = None
    try:
        with (run/'fit.log').open('wb') as log:
            proc = subprocess.Popen(command(candidate, run, dataset, python, 'fit', chip),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=train_timeout)
            except subprocess.TimeoutExpired as e:
                raise CandidateFailure('fitting budget exceeded') from e
            finally:
                kill(proc)
        if code:
            raise CandidateFailure(f'fit exited {code}: ' + (run/'fit.log').read_text(errors='replace')[-3000:])
        result['fit_seconds'] = time.monotonic()-started
        artifacts = artifact_hashes(run/'artifacts')
        predictor = Predictor(command(candidate, run, dataset, python, 'predict', chip), run, reward_timeout)
        calibration = DataView(dataset/'fit/val')
        audit_maps = next(calibration.iter_batches(batch_size=4, shuffle=False, seed=9201))['maps']
        try:
            result['independence'] = check_independence(predictor, audit_maps)
        except ValueError as exc:
            raise CandidateFailure(str(exc)) from exc
        result.update(score_development(predictor, dataset, max_ind=max_ind))
        result.update(correctness=1, artifacts=artifacts)
        if artifact_hashes(run/'artifacts') != artifacts:
            raise CandidateFailure('prediction mutated fitted detector artifacts')
        if bundle_identity(candidate, config, manifest) != identity:
            raise CandidateFailure('candidate/runtime changed during evaluation')
    except CandidateFailure as e:
        result.update(correctness=0, candidate_error=str(e), reward=None)
    finally:
        if predictor is not None:
            predictor.close()
    result['wall_seconds'] = time.monotonic()-started
    (run/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def score_development(predict, dataset, max_ind=3232):
    from .vendor.fairdiscover.shifts import FAMILIES, proxy_shift
    from .vendor.fairdiscover.metric import per_family_scores
    view = DataView(dataset/'judge/development')
    ids = np.random.default_rng(78231).permutation(len(view))[:max_ind]
    # Noise/proxy randomness is fixed by map identity, independent of batching and detector seed.
    scores = {name: [] for name in ['ind', *FAMILIES]}
    for start in range(0, len(ids), 16):
        selected = ids[start:start+16]
        clean = view.clean(selected)
        examples, keys = [], []
        for j, index in enumerate(selected):
            for fi, family in enumerate(['ind', *FAMILIES]):
                rng = np.random.default_rng(np.random.SeedSequence([71931, int(index), fi]))
                signal_map = clean[j] if family == 'ind' else proxy_shift(clean[j], view.mask, family, rng, 1.)
                noisy = (signal_map + rng.normal(0, NOISE_SIGMA, signal_map.shape)) * view.mask
                examples.append(noisy.astype(np.float32)); keys.append(family)
        # Hide family blocks and call sequence from the candidate.
        order = np.random.default_rng(81003+start).permutation(len(examples))
        for first in range(0, len(order), 32):
            ix = order[first:first+32]
            batch_scores = predict(np.stack([examples[i] for i in ix]))
            for i, score in zip(ix, batch_scores): scores[keys[i]].append(float(score))
    per_family = per_family_scores(scores['ind'], {k:v for k,v in scores.items() if k != 'ind'})
    return dict(proxy_score=float(np.mean(list(per_family.values()))), per_family=per_family,
                n_ind=len(ids), tail_resolution=1/len(ids), score_samples=scores)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--owner-pid', type=int)
    p.add_argument('--owner-start')
    for name in ('candidate', 'dataset', 'run', 'python'): p.add_argument('--'+name, required=True)
    for name, default in [('chip',0),('seed',0),('steps',250),('batch-size',16),('max-ind',3232)]:
        p.add_argument('--'+name, type=int, default=default)
    a = p.parse_args()
    args = vars(a)
    owner_pid, owner_start = args.pop('owner_pid'), args.pop('owner_start')
    if bool(owner_pid) != bool(owner_start):
        p.error('owner PID and identity must be supplied together')
    if owner_pid:
        from tpu.science.worker import watch_owner
        watch_owner(owner_pid, owner_start)
    print(json.dumps(evaluate(**args), allow_nan=False))
