"""Export figures comparing fixed methods on new-noise synthetic diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(a):
    root,out=Path(a.root),Path(a.out);out.mkdir(parents=True,exist_ok=False)
    closure=json.loads((root/'closure-comparison-001/report.json').read_text())
    calibration=json.loads((root/'score-calibration-001/report.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    methods=('06-grid-hybrid','scaled-hybrid','07-grid-spatial','scaled-spatial')
    labels=('06: existing hybrid','08: corrected hybrid','07: existing + spatial','09: corrected + spatial')
    colors=('#6689b9','#174c81','#b9ad7d','#81721b')
    probes=['spectral_low:0.2','spectral_high:0.2','quadratic_peak:0.2','quadratic_peak:-0.2',
            'anisotropic:0.2','gaussian_mixture:0.5','phase_iaaft:1.0','spatial_modulation:0.5']
    names=['Smoothing','Sharpening','Enhanced\npeaks','Reduced\npeaks','Anisotropy','Gaussian\nmixture','IAAFT\nstructure','Spatial\nmodulation']
    fig,ax=plt.subplots(figsize=(12,5.4),layout='constrained')
    for i,(method,label,color) in enumerate(zip(methods,labels,colors)):
        v=[closure['results'][method]['probes'][p] for p in probes]
        ax.bar(np.arange(len(probes))+(i-1.5)*.2,v,width=.19,color=color,label=label)
    ax.set(xticks=np.arange(len(probes)),xticklabels=names,ylim=(0,1),
           ylabel='Mean TPR, FPR 0.001–0.05 (higher is better)')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,ncols=2,loc='upper center')
    ax.set_title('Frozen correction tested with fresh noise',loc='left',fontsize=16,weight='bold')
    fig.supxlabel('404 reused audit maps per synthetic family; fresh measurement noise and float16 storage. NOT competition scores.',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(out/('r4-fresh-noise.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for ax,names_,title in zip(axes,[('continuous-summary','scaled-summary'),('continuous-hybrid','scaled-hybrid')],
                              ['Summary score','Summary + field hybrid']):
        for name,label,color in zip(names_,['Before normal correction','After normal correction'],['#888888','#174c81']):
            bins=calibration['known_parameter_false_positives'][name]['Omega_m']
            n=np.asarray([v['count'] for v in bins]);p=np.asarray([v['fraction'] for v in bins]);z=1.96
            center=(p+z*z/(2*n))/(1+z*z/n);width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            x=np.arange(1,6);ax.plot(x,100*p,'o-',color=color,label=label)
            ax.fill_between(x,100*(center-width),100*(center+width),color=color,alpha=.1)
        ax.axhline(1,color='black',linestyle='--',linewidth=.8)
        ax.set(title=title,xlabel='Matter-density quintile',ylabel='Normal maps flagged (%)',xticks=range(1,6),ylim=(0,6))
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Normal correction reduces, but does not eliminate, cosmology dependence',fontsize=13,weight='bold')
    fig.supxlabel('Previously used audit noise. Per-bin descriptive Wilson intervals; calibration and model-selection uncertainty excluded.',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(out/('r4-normal-coverage.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);main(p.parse_args())
