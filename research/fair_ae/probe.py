"""Uniform normal-only linear probes and representation diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--run',required=True);a=p.parse_args()
run=Path(a.run);root=Path(a.raw)
y=np.load(root/'label_newrealization.npy',allow_pickle=False).reshape((-1,5))
order=np.random.default_rng(2026).permutation(256)
ids={k:np.array([c*256+r for c in range(101) for r in v]) for k,v in [('train',order[:160]),('val',order[160:192]),('calibration',order[192:224]),('audit',order[224:])]}
z={k:np.load(run/'density'/f'{k}_latent.npy',allow_pickle=False).astype(float) for k in ids}
xm=z['train'].mean(0);xs=np.maximum(z['train'].std(0),1e-6)
x={k:(v-xm)/xs for k,v in z.items()}
ym=y[ids['train']].mean(0);ys=np.maximum(y[ids['train']].std(0),1e-8)
t={k:(y[v]-ym)/ys for k,v in ids.items()}
a=x['train'];g=a.T@a/len(a);b=a.T@t['train']/len(a)
choices=[];ws=[]
for ridge in (.001,.01,.1,1.):
 w=np.linalg.solve(g+ridge*np.eye(g.shape[0]),b);v=((x['val']@w-t['val'])**2).mean(0)
 choices.append(dict(ridge=ridge,val_normalized_mse=v.tolist()));ws.append(w)
best=np.argmin([np.mean(c['val_normalized_mse'][:2]) for c in choices]);w=ws[best]
e=np.linalg.eigvalsh(g);rank=float(e.sum()**2/max(np.square(e).sum(),1e-12))
report=dict(kind='normal_linear_probe_NOT_OOD',latent_dim=len(xm),correlation_effective_rank=rank,selected=choices[best],choices=choices,audit_normalized_mse=((x['audit']@w-t['audit'])**2).mean(0).tolist())
(run/'linear_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
