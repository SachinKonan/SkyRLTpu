"""Standalone research figures. All anomaly labels shown here are synthetic."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare import read_npz, inspect
from check_noise_hybrid import groups
from common import load_labels, write_json


def main(a):
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=False)
    final = root / 'final-comparison-001'; spatial = root / 'regional-calibration-001'
    models = {
        'Original\nsummary': read_npz(final / '04-summary.npz'),
        'Original\nhybrid': read_npz(final / '05-summary-field.npz'),
        'Noise-trained\nsummary': read_npz(final / 'noise-long-summary.npz'),
        'Noise-trained\nhybrid': read_npz(final / 'noise-long-field.npz'),
        'Regional\nspecialist': read_npz(spatial / 'regional-centered.npz'),
        'Hybrid +\nspecialist (max)': read_npz(spatial / 'hybrid+centered-max.npz')}
    reports = {k: inspect(v) for k, v in models.items()}
    probes = {
        'Low-frequency shift, 0.2': 'development/spectral_low:0.2',
        'High-frequency shift, 0.2': 'development/spectral_high:0.2',
        'Positive peaks, 0.2': 'development/quadratic_peak:0.2',
        'Negative peaks, 0.2': 'morphology_r2/negative_peak:0.2',
        'Randomized structure (IAAFT)': 'morphology_r2/phase_iaaft:1.0',
        'Anisotropic smoothing, 0.2': 'r1_novel_now_reused/anisotropic:0.2',
        'Spatial modulation, 0.5': 'r1_novel_now_reused/spatial_modulation:0.5'}
    matrix = np.asarray([[r['probes'][key] for r in reports.values()] for key in probes.values()])
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(12.5, 6.9), layout='constrained')
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(models)), labels=list(models)); ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, pad=10)
    ax.set_yticks(np.arange(len(probes)), labels=list(probes))
    for i in range(len(probes)):
        for k in range(len(models)):
            ax.text(k, i, f'{matrix[i,k]:.3f}', ha='center', va='center',
                    color='white' if matrix[i,k] > .5 else '#13212b', fontsize=12)
    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(probes), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2); ax.tick_params(which='minor', length=0)
    ax.set_title('Complementary information creates tradeoffs', loc='left', pad=58, fontsize=17, weight='bold')
    fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label='Mean TPR, FPR 0.001–0.05 (higher is better)')
    fig.supxlabel('Reused synthetic diagnostics — NOT competition scores. Model selection uncertainty is not shown.', fontsize=10)
    for suffix in ('png', 'pdf'): fig.savefig(out / ('r3-tradeoffs.' + suffix), dpi=180)
    plt.close(fig)
    # Normal-data parameter checks: no synthetic or public labels.
    labels = load_labels(a.raw)
    names = ('Original\nhybrid', 'Noise-trained\nhybrid', 'Hybrid +\nspecialist (max)')
    colors = ('#3465a4', '#ed7d31', '#4f9c71')
    stats = {name: groups(models[name], labels) for name in names}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout='constrained')
    for ax, parameter in zip(axes.flat, ('Omega_m', 'S8', 'T_AGN', 'f0', 'delta_z')):
        for name, color in zip(names, colors):
            bins = stats[name][parameter]; n = np.asarray([v['count'] for v in bins]); p = np.asarray([v['fraction'] for v in bins])
            z = 1.96; center = (p+z*z/(2*n))/(1+z*z/n)
            width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            x = np.arange(1, len(p)+1)
            ax.plot(x, 100*p, 'o-', color=color, label=name.replace('\n',' '), linewidth=1.5, markersize=4)
            ax.fill_between(x, 100*(center-width), 100*(center+width), color=color, alpha=.08)
        ax.axhline(1, color='#777777', linestyle='--', linewidth=1)
        ax.set(xlabel='Parameter quintile (training boundaries)', ylabel='Normal maps flagged (%)', title=parameter,
               xticks=np.arange(1,6), ylim=(0,8)); ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    handles, names_display = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, names_display, loc='center', frameon=False)
    fig.suptitle('A global 1% threshold is not conditionally uniform', fontsize=16, weight='bold')
    fig.supxlabel('Reused normal audit maps; bands are descriptive 95% Wilson intervals per bin. No multiple-testing adjustment.', fontsize=9)
    for suffix in ('png','pdf'): fig.savefig(out / ('r3-normal-coverage.' + suffix), dpi=180)
    plt.close(fig)
    write_json(out / 'normal-parameter-bins.json', stats)
    if (root / 'spatial-generalization-001/report.json').exists():
        r = json.loads((root / 'spatial-generalization-001/report.json').read_text())['results']
        methods = ('04-summary','noise-long-summary','regional-raw','regional-centered')
        patterns = ('sine_original','sine_random_phase','sine_two_cycles','linear_gradient')
        names = ('Original sine\nfresh noise','Random phase\none cycle','Random phase\ntwo cycles','Linear\ngradient')
        fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
        for offset, method, color in zip(np.arange(4)-1.5, methods, ('#3465a4','#ed7d31','#9d7fbd','#4f9c71')):
            values = [r[method]['scores'][key] for key in patterns]
            ax.bar(np.arange(4)+offset*.2, values, width=.19, label=method, color=color)
        ax.set(xticks=np.arange(4), xticklabels=names, ylabel='Mean TPR, FPR 0.001–0.05', ylim=(0,1))
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.legend(frameon=False, loc='upper right')
        ax.set_title('Frozen specialist on fresh noise and new patterns', loc='left', fontsize=16, weight='bold')
        fig.supxlabel('1024 reused normal audit realizations. New synthetic patterns; no refitting. NOT competition scores.', fontsize=9)
        for suffix in ('png','pdf'): fig.savefig(out / ('r3-pattern-generalization.' + suffix), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','raw','out'): p.add_argument('--'+name,required=True)
    main(p.parse_args())
