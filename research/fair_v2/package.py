"""Package a fitted detector combination as a Codabench Phase-2 code submission.

  python package.py --run <fit_v3 run dir> --detectors Sd,C,C1 --cnn cnn0=<net.pt> cnn1=<net.pt> --out <dir>

Writes <dir>/{model.py, fair_stats.py, fair_cnn.py, fair_lik.py, mask.npy, edges.npz, cnn*.pt, det_<tag>.pt,
combo.json} and <dir>.zip. Model.predict(test_data) scores every map independently (fixed-shape batches, eval mode).
"""
import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from groups import groups

HERE = Path(__file__).resolve().parent


def export_detector(path, tag):
    m = pickle.load(open(path, 'rb'))
    lk, std = m['lk'], m['std']
    G, _ = groups()
    sources = []
    for item in m['spec'].split('|'):
        src, subset = item.split(':')
        sources.append((src, subset if subset == 'raw' else [int(i) for i in
                                                           np.concatenate([G[g] for g in subset.split('+')])]))
    cpu = lambda t: t.detach().cpu() if torch.is_tensor(t) else t
    net = lk.net
    width = net.mu.in_features
    depth = sum(isinstance(l, torch.nn.Linear) for l in net.body)
    blob = dict(
        tag=tag, sources=sources, stat=m['stat'],
        f_mean=torch.as_tensor(lk.f_mean), f_std=torch.as_tensor(lk.f_std), keep=torch.as_tensor(lk.keep),
        t_mean=torch.as_tensor(lk.t_mean), t_std=torch.as_tensor(lk.t_std),
        t_lo=torch.as_tensor(lk.t_lo), t_hi=torch.as_tensor(lk.t_hi), cosmo=cpu(lk.cosmo), hull_r=lk.hull_r,
        width=width, depth=depth, dk=int(lk.keep.sum()),
        net={k: cpu(v) for k, v in net.state_dict().items()},
        reg={k: cpu(v) for k, v in lk.reg.state_dict().items()},
        Rinv=cpu(lk.Rinv), Rc_inv=cpu(lk.Rc_inv), cq=cpu(lk.cq), cg=cpu(lk.cg),
        std_loc=torch.as_tensor(std.loc), std_scale=torch.as_tensor(std.scale), std_norm=std.norm)
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--detectors', required=True, help='comma list of fit_v3 tags, combined by mean of z')
    ap.add_argument('--cnn', nargs='*', default=[], help='source=path/to/net.pt')
    ap.add_argument('--edges', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')
    ap.add_argument('--mask', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/WIDE12H_bin2_2arcmin_mask.npy')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for src, dst in (('stats.py', 'fair_stats.py'), ('cnn_infer.py', 'fair_cnn.py'), ('lik_infer.py', 'fair_lik.py'),
                     ('model_template.py', 'model.py')):
        shutil.copy(HERE / 'pack' / src, out / dst)
    shutil.copy(args.mask, out / 'WIDE12H_bin2_2arcmin_mask.npy')
    shutil.copy(args.edges, out / 'edges.npz')
    cnns = {}
    for item in args.cnn:
        src, path = item.split('=')
        st = torch.load(path, map_location='cpu', weights_only=False)
        torch.save(dict(net=st['net'], widths=list(st.get('widths', (32, 64, 128, 192, 256, 256)))), out / f'{src}.pt')
        cnns[src] = f'{src}.pt'
    dets = []
    for tag in args.detectors.split(','):
        blob = export_detector(Path(args.run) / f'model_{tag}.pkl', tag)
        torch.save(blob, out / f'det_{tag}.pt')
        dets.append(f'det_{tag}.pt')
    (out / 'combo.json').write_text(json.dumps(dict(detectors=dets, cnns=cnns, rule='mean'), indent=1))
    shutil.make_archive(str(out), 'zip', root_dir=out)
    print('packaged', out, 'and', str(out) + '.zip')


if __name__ == '__main__':
    main()
