"""Record the negative R5 finding and plot its matched controls."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(a):
    root = Path(a.root)
    comparison = json.loads((root/'comparison-001/report.json').read_text())
    controls = json.loads((root/'matched-controls-001/report.json').read_text())
    probes = [('development/spectral_low:0.2','Smoothing'),
              ('development/spectral_high:0.2','Sharpening'),
              ('development/quadratic_peak:0.2','Stronger peaks'),
              ('morphology_r2/negative_peak:0.2','Weaker peaks'),
              ('morphology_r2/phase_iaaft:1.0','Phase randomization')]
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    panels=[(comparison['results'], [('raw_energy','All-coordinate energy'),
                                   ('diagonal_max','Largest coordinate'),
                                   ('diagonal_bj','Berk-Jones-style statistic')],
             'Concentrating on extreme coordinates loses sensitivity'),
            (controls['results'], [('08','Current ensemble (08)'),
                                  ('08+diagonal_bj-mean','Add sparse statistic'),
                                  ('08+raw_energy-mean','Matched dense-score control')],
             'The phase-randomization gain is also obtained by reweighting existing scores')]
    x=np.arange(len(probes));width=.25
    colors=['#176d96','#e87936','#4a9776']
    for ax,(reports,models,title) in zip(axes,panels):
        for i,((key,label),color) in enumerate(zip(models,colors)):
            y=[reports[key]['probes'][p] for p,_ in probes]
            ax.bar(x+(i-1)*width,y,width,label=label,color=color)
        ax.set_ylim(0,.95);ax.set_ylabel('Mean TPR');ax.set_title(title,loc='left',fontsize=11)
        ax.legend(loc='upper left',fontsize=9,ncol=1);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xticks(x,[label for _,label in probes],fontsize=10)
    fig.suptitle('R5: sparse latent statistics did not earn a new submission candidate',fontsize=14)
    fig.text(.5,.015,'Reused synthetic diagnostics; mean TPR over 0.1%–5% FPR. These are not Codabench results.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.95));figures=root/'figures-001';figures.mkdir(exist_ok=False)
    for ext in ('png','pdf'):fig.savefig(figures/('r5-matched-controls.'+ext),dpi=160)
    result=dict(round='R5',date='2026-10-09',goal_status='Active; official performance unmeasured.',
        finding='No R5 method is promoted. Sparse coordinate scores usually reduce sensitivity. The IAAFT ensemble improvement is matched by adding more of the existing dense summary score.',
        sources=[dict(url='https://arxiv.org/abs/math/0410072',role='Rare/weak sparse signal motivation; its guarantees do not apply automatically to learned dependent coordinates.'),
                 dict(url='https://arxiv.org/abs/math/0603238',role='Berk-Jones/phi-divergence statistic family.'),
                 dict(url='https://arxiv.org/abs/1311.3190',role='Distinguish the KL-supremum statistic from exact Berk-Jones M statistics and exact p-values.')],
        measurements=dict(latent_diagnostics=comparison['latent_diagnostics'],
                          gaussian_control=json.loads((root/'gaussian-statistics-check.json').read_text()),
                          IAAFT_sparse_minus_matched_dense=controls['bj_minus_matched_dense_mean']['morphology_r2/phase_iaaft:1.0'],
                          rotation_audit_rank_correlations=controls['audit_rank_correlations_original_vs_rotated_max'],
                          selected_results={name:{p:report['probes'][p] for p,_ in probes}
                                            for name,report in controls['results'].items()},
                          normal_audit_1pct_counts={name:report['normal']['exceedances'][1]
                                                   for name,report in comparison['results'].items()}),
        evidence=dict(latent_archive=json.loads((root/'latent-cache-001.receipt.json').read_text()),
                      initial_report=str(root/'comparison-001/report.json'),
                      matched_control_report=str(root/'matched-controls-001/report.json'),
                      all_initial_scores=str(root/'comparison-001/comparison.csv'),
                      all_control_scores=str(root/'matched-controls-001/comparison.csv')),
        verification=dict(extraction='Three frozen checkpoints; squared extracted latents reproduce their recorded energies.',
                          pointwise='All three extraction checks have zero singleton/order/chunk error.',
                          rotations='Four fixed orthogonal rotations preserve energy within numerical tolerance; none is selected.',
                          normal_reference='Coordinate moments fit normal validation only. Calibration supplies score scales and thresholds.'),
        limitations=['All diagnostic families and clean maps were previously examined.',
                     'The follow-up dense and rotation controls were added after seeing the initial R5 results.',
                     'Bootstrap intervals condition on the fixed models and empirical IID samples; they do not correct model selection or shared simulation dependence.',
                     'Latent-axis failure does not rule out sparse anomalies in physically meaningful coordinates.',
                     'Candidates 04 through 09 remain unchanged; no public test reads and no competition submissions in R5.'],
        next_hypothesis='R6: test the field flow terminal noise for non-Gaussianity and spatial dependence, inspired by arXiv:2605.22496; numerical/normal-data pilot first.')
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(out=a.out,figures=str(figures),decision=result['finding'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
