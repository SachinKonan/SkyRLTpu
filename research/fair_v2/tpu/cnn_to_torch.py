"""Convert TPU-trained JAX CNNs (params.npz) to torch cnn.Net checkpoints and verify vs ref.npz."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import Net  # noqa: E402


def convert(run):
    run = Path(run)
    args = json.loads((run / 'log.json').read_text())['args']
    widths = tuple(args['widths'])
    z = np.load(run / 'params.npz')
    P = {k[2:]: z[k] for k in z.files if k.startswith('p/')}
    S = {k[2:]: z[k] for k in z.files if k.startswith('s/')}
    net = Net(widths)
    sd = {}
    conv = lambda a: torch.as_tensor(np.ascontiguousarray(a.transpose(3, 2, 0, 1)))
    dense = lambda a: torch.as_tensor(np.ascontiguousarray(a.T))

    def bn(t, j):
        sd[f'{t}.weight'] = torch.as_tensor(P[f'{j}/scale']); sd[f'{t}.bias'] = torch.as_tensor(P[f'{j}/bias'])
        sd[f'{t}.running_mean'] = torch.as_tensor(S[f'{j}/mean']); sd[f'{t}.running_var'] = torch.as_tensor(S[f'{j}/var'])
        sd[f'{t}.num_batches_tracked'] = torch.tensor(0)
    sd['stem.0.weight'] = conv(P['stem_conv/kernel']); bn('stem.1', 'stem_bn')
    for k in range(2 * len(widths)):
        sd[f'blocks.{k}.c1.weight'] = conv(P[f'blk{k}/c1/kernel']); bn(f'blocks.{k}.b1', f'blk{k}/b1')
        sd[f'blocks.{k}.c2.weight'] = conv(P[f'blk{k}/c2/kernel']); bn(f'blocks.{k}.b2', f'blk{k}/b2')
        if f'blk{k}/skip_conv/kernel' in P:
            sd[f'blocks.{k}.skip.0.weight'] = conv(P[f'blk{k}/skip_conv/kernel']); bn(f'blocks.{k}.skip.1', f'blk{k}/skip_bn')
    for t, j in (('emb.0', 'emb0'), ('emb.2', 'emb2'), ('head.1', 'head1')):
        sd[f'{t}.weight'] = dense(P[f'{j}/kernel']); sd[f'{t}.bias'] = torch.as_tensor(P[f'{j}/bias'])
    miss, extra = set(net.state_dict()) - set(sd), set(sd) - set(net.state_dict())
    assert not miss and not extra, (miss, extra)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}); net.eval()
    ref = np.load(run / 'ref.npz')
    mask = np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/WIDE12H_bin2_2arcmin_mask.npy')
    with torch.no_grad():
        pred, e = net(torch.as_tensor(ref['x']), torch.as_tensor(mask, dtype=torch.float32))
    torch.save(dict(net=net.state_dict(), widths=list(widths), t_mean=z['t_mean'], t_std=z['t_std'], source=str(run)),
               run / 'net_torch.pt')
    return float(np.abs(pred.numpy() - ref['pred']).max()), float(np.abs(e.numpy() - ref['emb']).max()), \
        float(np.abs(ref['emb']).max())


if __name__ == '__main__':
    for r in sys.argv[1:]:
        dp, de, s = convert(r)
        print(f'{r}: max |pred diff| {dp:.3e}  max |emb diff| {de:.3e} (emb scale {s:.2e})', flush=True)
