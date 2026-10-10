import json, sys
r = json.load(open(sys.argv[1]))
keys = ['bary9.0', 'bary6.5', 'baryDMO', 'peakcut0.3', 'ia0.2', 'poisson120', 'psfpat', 'srcclust0.15', 'supp0.15', 'boost0.15', 'aniso0.7', 'smooth0.5', 'noisegrad0.15', 'gradient']
sel = sys.argv[2].split(',') if len(sys.argv) > 2 else list(r)
print('| model | dims | normal q99 / q99.5 / q99.9 | shift_mean (19) | ' + ' | '.join(keys) + ' |')
print('|' + '---|' * (4 + len(keys)))
for k in sel:
    v = r[k]; q = v['normal_quantiles']
    print(f"| {k} | {v.get('dims', '')} | {q[3]:.2f} / {q[4]:.2f} / {q[5]:.2f} | {v['shift_mean']:.4f} | " + ' | '.join(f"{v['shifts'].get(s, float('nan')):.3f}" for s in keys) + ' |')
