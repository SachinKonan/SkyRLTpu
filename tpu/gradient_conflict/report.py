"""Produce a compact report and an exportable plot from completed probe groups."""

import argparse
import json
from pathlib import Path

import numpy as np


def report(source, output_dir):
    data = json.loads(Path(source).read_text())
    groups = data['groups']
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Qwen valid versus invalid gradient pilot', '',
        f"Completed groups: {len(groups)} / 6. Full run complete: {data.get('complete', False)}.", '',
        f"Frozen checkpoint: `{data['checkpoint']}`; rank-{data['lora_rank']} LoRA, "
        f"{data.get('accelerator', 'tpu-v4-32')} "
        f"TP{data.get('mesh', {}).get('tp', 8)}/FSDP{data.get('mesh', {}).get('fsdp', 2)}.", '',
        'These are text-reconstructed, signed policy gradients at one checkpoint. '
        'Historical sampling logprobs, original token IDs, and phase-two action masks were unavailable. '
        'The six selected groups describe the observed validity range, rather than a random sample of all training batches.', '',
        '| Group | Valid / total | Valid sum norm | Invalid sum norm | Invalid / valid norm | Cosine | Random-split cosine | Cancellation ratio |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    def fmt(x): return 'undefined' if x is None else f'{x:.5g}'
    for group in groups:
        s = group['valid_vs_invalid']
        lines.append(f"| {group['id']} | {s['left_count']} / {s['left_count']+s['right_count']} | "
                     + ' | '.join(fmt(s[k]) for k in ['left_sum_norm','right_sum_norm','right_to_left_norm','cosine'])
                     + f" | {fmt(group['random_split']['cosine'])} | {fmt(s['cancellation_ratio'])} |")
    lines += ['', 'Sum norms measure population contributions. Mean-gradient norms, token counts, per-leaf '
              'comparisons, and signed dot products with the combined gradient are retained in `result.json`.', '']
    cosines = [g['valid_vs_invalid']['cosine'] for g in groups if g['valid_vs_invalid']['cosine'] is not None]
    if cosines:
        lines += [f"Negative cosine in {sum(c < 0 for c in cosines)} / {len(cosines)} nonzero comparisons; "
                  f"median cosine {np.median(cosines):.5g} (range {min(cosines):.5g} to {max(cosines):.5g}).", '']
    if groups:
        valid_adverse = sum(g['valid_vs_invalid']['combined_dot_left'] < 0 for g in groups)
        invalid_adverse = sum(g['valid_vs_invalid']['combined_dot_right'] < 0 for g in groups)
        lines += [
            f"The combined gradient has a negative dot product with the valid gradient in {valid_adverse} / {len(groups)} groups, "
            f"and with the invalid gradient in {invalid_adverse} / {len(groups)} groups. "
            'A sufficiently small SGD step along the negative combined gradient would increase the corresponding '
            'subset loss when this dot product is negative.', '',
            'All selected valid samples have positive original advantages; all selected invalid samples have negative '
            'advantages. These are signed training contributions: rewarding valid responses versus suppressing invalid '
            'responses. Their cosine is not the cosine of two unsigned likelihood gradients. '
            'The random partition is one matched-size control per group, not a statistical null distribution.', '',
        ]
    for g in groups:
        if g.get('reconstruction_status') in ('pending', 'failed'):
            lines += [f"Full-group reconstruction check: **{g['reconstruction_status']}**. Interpret this partial result accordingly.", '']
        if 'reconstruction_relative_error' in g:
            lines += [f"Independent full-group reconstruction relative error: {g['reconstruction_relative_error']:.6g}.", '']
    lines += [f"Parameter fingerprint across drains: `{data.get('parameter_sha256', 'not yet available')}`.", '',
              'A negative cosine demonstrates local opposition of signed loss gradients. It does not by itself '
              'establish harm to task reward or predict an Adam update. No model update was applied.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    (out/'result.json').write_text(json.dumps(data,indent=2)+'\n')
    if not groups:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.array([g['valid_fraction'] for g in groups]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x, [g['valid_vs_invalid']['cosine'] for g in groups], 'o-', label='Valid vs invalid')
    axes[0].plot(x, [g['random_split']['cosine'] for g in groups], 's--', label='Matched random split')
    axes[0].axhline(0, color='gray', lw=1)
    axes[0].set(xlabel='Valid samples (%)', ylabel='Gradient cosine', ylim=(-1.05, 1.05))
    axes[0].legend()
    axes[1].plot(x, [g['valid_vs_invalid']['right_to_left_norm'] for g in groups], 'o-')
    axes[1].axhline(1, color='gray', lw=1)
    axes[1].set(xlabel='Valid samples (%)', ylabel='Invalid / valid sum-gradient norm')
    fig.suptitle('Qwen3.5-27B frozen LoRA gradient diagnostic')
    fig.savefig(out/'gradient-conflict.png', dpi=180)
    fig.savefig(out/'gradient-conflict.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    report(args.source, args.output_dir)
