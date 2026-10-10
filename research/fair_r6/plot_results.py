"""Show every registered R6 diagnostic and its normal false-alarm count."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(a):
    report=json.loads(Path(a.report).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    labels={'energy':'Latent energy','ad':'Normality (AD)','ad_shape':'Distribution shape',
            'psd_cv':'Spatial spectrum','correlation_energy':'Spatial correlation',
            'sitn-inspired':'Normality + spectrum','shape-spatial':'Shape + spatial structure',
            '08':'Current ensemble 08','09':'Current spatial ensemble 09',
            '08+sitn-inspired-mean':'08 + normality/spectrum (mean)',
            '08+sitn-inspired-max':'08 + normality/spectrum (max)',
            '08+shape-spatial-mean':'08 + shape/spatial (mean)',
            '08+shape-spatial-max':'08 + shape/spatial (max)'}
    probes=[('spectral_low:0.2','Smoothing'),('spectral_high:0.2','Sharpening'),
            ('quadratic_peak:0.2','Stronger\npeaks'),('quadratic_peak:-0.2','Weaker\npeaks'),
            ('gaussian_mixture:0.5','Gaussian\nmixing'),('anisotropic:0.2','Anisotropy'),
            ('spatial_modulation:0.5','Spatial\nmodulation'),('phase_iaaft:1.0','Phase\nrandomization')]
    names=list(labels);values=np.array([[report['results'][n]['probes'][p] for p,_ in probes] for n in names])
    fig,(ax,side)=plt.subplots(1,2,figsize=(14,8),gridspec_kw=dict(width_ratios=[8,2]),sharey=True)
    heat=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_yticks(np.arange(len(names)),[labels[n] for n in names],fontsize=9)
    ax.set_xticks(np.arange(len(probes)),[label for _,label in probes],fontsize=9)
    ax.xaxis.tick_top();ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
    for row in range(len(names)):
        for col in range(len(probes)):
            ax.text(col,row,f'{values[row,col]:.3f}',ha='center',va='center',fontsize=9,
                    color='black' if values[row,col]>.55 else 'white')
    fp=np.array([report['results'][n]['normal']['normal_float16']['exceedances'][1]/3232*100 for n in names])
    side.barh(np.arange(len(names)),fp,color='#4f83a8');side.axvline(1,color='#ad3a32',ls='--',lw=1)
    side.set_xlabel('Normal maps flagged (%)');side.set_title('Nominal 1% FPR',fontsize=10)
    side.tick_params(axis='y',left=False,labelleft=False);side.grid(axis='x',alpha=.2);side.set_axisbelow(True)
    side.set_xlim(0,max(1.8,float(fp.max())*1.15))
    for i,v in enumerate(fp):side.text(v+.02,i,f'{v:.2f}',va='center',fontsize=8)
    fig.suptitle('R6 field-noise diagnostics: mean TPR at 0.1%–5% false-positive rates',fontsize=14)
    fig.text(.5,.025,'Reused normal maps and synthetic families; not competition results. All 13 registered comparisons are shown.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.255,right=.97,top=.86,bottom=.21,wspace=.14)
    cax=fig.add_axes([.38,.11,.39,.018])
    fig.colorbar(heat,cax=cax,orientation='horizontal',label='Mean TPR')
    for ext in ('png','pdf'):fig.savefig(out/('r6-diagnostics.'+ext),dpi=160)
    print(json.dumps(dict(figures=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
