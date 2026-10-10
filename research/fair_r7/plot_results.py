"""Show joint-density comparisons and the explicitly post-result radial control."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(a):
    root,out=Path(a.root),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    first=json.loads((root/'comparison-001/report.json').read_text())
    radial=json.loads((root/'radial-controls-001/report.json').read_text())
    probes=[('spectral_low:0.2','Smoothing'),('spectral_high:0.2','Sharpening'),
            ('quadratic_peak:0.2','Stronger\npeaks'),('quadratic_peak:-0.2','Weaker\npeaks'),
            ('gaussian_mixture:0.5','Gaussian\nmixing'),('anisotropic:0.2','Anisotropy'),
            ('spatial_modulation:0.5','Spatial\nmodulation'),('phase_iaaft:1.0','Phase\nrandomization')]
    entries=[(first,'08','Current hybrid 08'),(first,'09','Current spatial hybrid 09'),
             (first,'hybrid/oof/diagonal','Independent Gaussian (squared)'),
             (first,'hybrid/oof/gaussian_joint','Joint Gaussian (squared)'),
             (first,'hybrid/oof/kde_independent','Independent KDE'),
             (first,'hybrid/oof/kde_joint','Joint KDE'),
             (radial,'hybrid/oof/diagonal','Independent Gaussian radius *'),
             (radial,'hybrid/oof/gaussian_joint','Joint Gaussian radius *')]
    values=np.array([[r['results'][k]['probes']['closure/'+p] for p,_ in probes] for r,k,_ in entries])
    fig,ax=plt.subplots(figsize=(13,6.5));heat=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_yticks(range(len(entries)),[x[2] for x in entries],fontsize=10)
    ax.set_xticks(range(len(probes)),[x[1] for x in probes],fontsize=9);ax.xaxis.tick_top()
    for row in range(len(entries)):
        for col in range(len(probes)):
            ax.text(col,row,f'{values[row,col]:.3f}',ha='center',va='center',fontsize=10,
                    color='black' if values[row,col]>.55 else 'white')
    ax.axhline(5.5,color='white',lw=2)
    fig.suptitle('R7: joint likelihood statistics do not give a broad improvement',fontsize=14)
    fig.text(.5,.055,'Mean TPR at 0.1%–5% FPR on reused synthetic diagnostics; these are not competition scores.',ha='center',fontsize=10)
    fig.text(.5,.021,'* Exploratory control added after the registered result: square root preserves standalone rankings, but changes ensemble weighting.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.28,right=.96,top=.84,bottom=.23)
    cax=fig.add_axes([.38,.14,.43,.018]);fig.colorbar(heat,cax=cax,orientation='horizontal',label='Mean TPR')
    for ext in ('png','pdf'):fig.savefig(out/('r7-joint-statistics.'+ext),dpi=160)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);main(p.parse_args())
