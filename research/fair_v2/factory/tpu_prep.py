"""Build and upload the TPU grading bundle for tpu_grader (della side, CPU).

  python factory/tpu_prep.py [--stack v1 v2] [--skip-noisy] [--extra venv.tar.gz=<sha256>] [--upload]

Local bundle (fair-data/factory/tpu_bundle/v1/), mirrored to gs://sk7524-tinker-tpu-us-east5/fair-factory/v1/.
Every path is relative to /dev/shm/fair on the TPU host (tpu/fair/ray_task.py stage() consumes manifest.json):
  factory/noisy/{fit_v0,fit_v1,calib_v0,calib_v1,audit_v0}.npy   frozen noisy dumps, all rows (float16)
  factory/noisy/shift_<s>_v0.npy                                  frozen shift dumps, held-out audit rows only
  factory/meta.npz        labels, per-fold masks, held audit rows, footprint mask, shift order
  factory/cache_<stack>/fold{0,1,2}.npz + stack.json             stack members + consRplus views per fold
  venv.tar.gz             the grader's Python env (venv/ + py/ interpreter), built on a TPU host (--extra)
  manifest.json           {"version", "files": [{"path", "sha256"}], "untar": [{"archive", "into"}]}
"""
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import lib  # noqa: E402
from holdout3 import masks  # noqa: E402
from tpu_grader import grade as TG  # noqa: E402

DUMPS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy')
BUNDLE = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/tpu_bundle/v1')
FAC = BUNDLE / 'factory'
CACHES = {'v1': Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/cache'),
          'v2': Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/cache_v2')}
GS = 'gs://sk7524-tinker-tpu-us-east5/fair-factory/v1'
GCLOUD = '/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud'
FOLDS = (0, 1, 2)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def build_meta():
    M = {f'labels_{n}': lib.labels(n) for n in ('fit', 'calib', 'audit')}
    held = np.zeros(len(M['labels_audit']), bool)
    for k in FOLDS:
        m = masks(k)
        for n in ('fit', 'calib', 'audit'):
            M[f'mask{k}_{n}'] = m[n]
        held |= m['audit']
    M['held_rows'] = np.nonzero(held)[0]
    M['mask'] = D.mask()
    M['shifts'] = np.array(lib.SHIFTS)
    return M


def write_caches(stacks):
    for v in stacks:
        # plain .npz (keys comp/<member>/<set>, TH/<view>/<set>) + stack.json: no pickle on the TPU side
        d = FAC / f'cache_{v}'
        d.mkdir(parents=True, exist_ok=True)
        for k in FOLDS:
            with open(CACHES[v] / f'fold{k}.pkl', 'rb') as fh:
                C = pickle.load(fh)
            arrs = {f'{part}/{m}/{n}': np.asarray(a) for part in ('comp', 'TH') for m, dd in C[part].items()
                    for n, a in dd.items()}
            np.savez(d / f'fold{k}.npz', **arrs)
        meta = CACHES[v] / 'stack.json'
        spec = json.loads(meta.read_text()) if meta.exists() else dict(stack=TG.STACK_V1, views=TG.VIEWS_V1)
        (d / 'stack.json').write_text(json.dumps(spec, indent=1))


def hashes():
    """sha256 of every bundle file, cached by (size, mtime) in .sha_cache.json."""
    cpath = BUNDLE / '.sha_cache.json'
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    out = {}
    for p in sorted(FAC.rglob('*')):
        if not (p.is_file() and p.suffix in ('.npy', '.npz', '.json')) or p.name.endswith('.tmp.npy'):
            continue
        rel = str(p.relative_to(BUNDLE))
        st = p.stat()
        key = [st.st_size, int(st.st_mtime)]
        if rel in cache and cache[rel][:2] == key:
            out[rel] = cache[rel][2]
        else:
            out[rel] = sha256(p)
            cache[rel] = key + [out[rel]]
    cpath.write_text(json.dumps(cache, indent=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack', nargs='+', default=['v1', 'v2'])
    ap.add_argument('--skip-noisy', action='store_true')
    ap.add_argument('--extra', nargs='*', default=[], help='path=sha256 entries uploaded separately (venv.tar.gz)')
    ap.add_argument('--upload', action='store_true')
    args = ap.parse_args()
    (FAC / 'noisy').mkdir(parents=True, exist_ok=True)
    M = build_meta()
    old = FAC / 'meta.npz'
    if not (old.exists() and all(np.array_equal(v, np.load(old)[k]) for k, v in M.items())):
        np.savez(old, **M)                               # rewrite only if changed (keeps its sha256 stable)
    held = M['held_rows']
    print(f'held audit rows {len(held)}; shifts {len(lib.SHIFTS)}', flush=True)
    if not args.skip_noisy:
        for n in ('fit_v0', 'fit_v1', 'calib_v0', 'calib_v1', 'audit_v0'):
            dst = FAC / 'noisy' / f'{n}.npy'
            if not dst.exists():
                dst.symlink_to(DUMPS / f'{n}.npy')
        for s in lib.SHIFTS:
            dst = FAC / 'noisy' / f'shift_{s}_v0.npy'
            if not dst.exists():
                a = np.load(DUMPS / f'shift_{s}_v0.npy', mmap_mode='r')
                np.save(dst.with_suffix('.tmp.npy'), np.ascontiguousarray(a[held]))
                dst.with_suffix('.tmp.npy').replace(dst)
    write_caches(args.stack)
    H = hashes()
    files = [dict(path=rel, sha256=h) for rel, h in H.items()]
    untar = []
    for e in args.extra:
        path, h = e.split('=')
        files.append(dict(path=path, sha256=h))
        if path.endswith('.tar.gz'):
            untar.append(dict(archive=path, into='.'))
    manifest = dict(version=f'fair-factory-v1-{time.strftime("%Y%m%dT%H%M%S")}', gcs=GS, files=files, untar=untar)
    (BUNDLE / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    size = sum((BUNDLE / f['path']).stat().st_size for f in files if (BUNDLE / f['path']).exists())
    print(f'{len(files)} files ({size/1e9:.1f} GB local), manifest {manifest["version"]}', flush=True)
    if args.upload:
        ls = subprocess.run([GCLOUD, 'storage', 'ls', '-l', '-r', f'{GS}/**'], capture_output=True, text=True).stdout
        have = {}
        for line in ls.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1].startswith(GS + '/'):
                have[parts[-1][len(GS) + 1:]] = int(parts[0])
        for f in files:
            p = BUNDLE / f['path']
            if not p.exists():
                assert f['path'] in have, f'{f["path"]} is neither local nor in the bucket'
                continue
            if have.get(f['path']) == p.stat().st_size and f['path'].endswith('.npy'):
                continue
            subprocess.run([GCLOUD, 'storage', 'cp', str(p.resolve()), f'{GS}/{f["path"]}'], check=True)
        subprocess.run([GCLOUD, 'storage', 'cp', str(BUNDLE / 'manifest.json'), f'{GS}/manifest.json'], check=True)
        print('uploaded', flush=True)


if __name__ == '__main__':
    main()
