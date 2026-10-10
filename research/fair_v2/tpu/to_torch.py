"""Convert a TPU-trained JAX U-Net (params.npz) to the torch UNet checkpoint format and verify it.

  python to_torch.py <run_dir> [<run_dir> ...]
Writes <run_dir>/net_torch.pt (dict(net=state_dict, widths=...)) and checks the torch forward pass
against <run_dir>/ref.npz (outputs computed on TPU for 4 maps with fixed noise).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agents' / 'denoiser'))
from unet import UNet  # noqa: E402


def convert(run):
    run = Path(run)
    args = json.loads((run / 'log.json').read_text())['args']
    widths = tuple(args['widths'])
    z = np.load(run / 'params.npz')
    P = {k[2:]: z[k] for k in z.files if k.startswith('p/')}
    S = {k[2:]: z[k] for k in z.files if k.startswith('s/')}
    net = UNet(widths)
    sd = {}
    conv = lambda a: torch.as_tensor(np.ascontiguousarray(a.transpose(3, 2, 0, 1)), dtype=torch.float32)

    def bn(tname, jname):
        sd[f'{tname}.weight'] = torch.as_tensor(P[f'{jname}/scale'])
        sd[f'{tname}.bias'] = torch.as_tensor(P[f'{jname}/bias'])
        sd[f'{tname}.running_mean'] = torch.as_tensor(S[f'{jname}/mean'])
        sd[f'{tname}.running_var'] = torch.as_tensor(S[f'{jname}/var'])
        sd[f'{tname}.num_batches_tracked'] = torch.tensor(0)

    def block(tname, jname):
        sd[f'{tname}.c1.weight'] = conv(P[f'{jname}/c1/kernel'])
        bn(f'{tname}.b1', f'{jname}/b1')
        sd[f'{tname}.c2.weight'] = conv(P[f'{jname}/c2/kernel'])
        bn(f'{tname}.b2', f'{jname}/b2')
        if f'{jname}/skip/kernel' in P:
            sd[f'{tname}.skip.weight'] = conv(P[f'{jname}/skip/kernel'])

    block('inc', 'inc')
    for i in range(len(widths) - 1):
        sd[f'down.{i}.0.weight'] = conv(P[f'down{i}_conv/kernel'])
        bn(f'down.{i}.1', f'down{i}_bn')
        block(f'enc.{i}', f'enc{i}')
    block('mid', 'mid')
    for j in range(len(widths) - 1):
        sd[f'up.{j}.weight'] = torch.as_tensor(P[f'up{j}/weight'])
        sd[f'up.{j}.bias'] = torch.as_tensor(P[f'up{j}/bias'])
        block(f'dec.{j}', f'dec{j}')
    sd['out.weight'] = conv(P['out/kernel'])
    sd['out.bias'] = torch.as_tensor(P['out/bias'])
    missing = set(net.state_dict()) - set(sd)
    extra = set(sd) - set(net.state_dict())
    assert not missing and not extra, (missing, extra)
    net.load_state_dict(sd)
    net.eval()
    ref = np.load(run / 'ref.npz')
    mask = np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/WIDE12H_bin2_2arcmin_mask.npy')
    with torch.no_grad():
        y, _ = net(torch.as_tensor(ref['x']), torch.as_tensor(mask, dtype=torch.float32))
    err = float(np.abs(y.numpy() - ref['y'])[:, mask].max())
    scale = float(np.abs(ref['y'][:, mask]).max())
    torch.save(dict(net=net.state_dict(), widths=list(widths), source=str(run)), run / 'net_torch.pt')
    return err, scale


if __name__ == '__main__':
    for r in sys.argv[1:]:
        e, s = convert(r)
        print(f'{r}: max |torch - jax| = {e:.3e} (output scale {s:.3e})', flush=True)
