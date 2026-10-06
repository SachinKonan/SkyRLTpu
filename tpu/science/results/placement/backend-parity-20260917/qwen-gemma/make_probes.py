import ast,copy
from pathlib import Path
p=Path('.science/placement-assessment-20260917/qwen-gemma-parity')
for model in ['qwen','gemma']:
 tree=ast.parse((p.parent/(model+'-best.py')).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='place')
 cost=next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name==('compute_cost' if model=='qwen' else 'cost_fn'))
 wire=copy.deepcopy(cost);wire.name='probe_wire'
 target='wire_cost' if model=='qwen' else 'wl_norm'
 end=next(i for i,n in enumerate(wire.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
 wire.body=wire.body[:end+1]+[ast.Return(ast.Name(target,ast.Load()))]
 end=fn.body.index(cost);fn.body=fn.body[:end+1]+[wire]
 extra='''
    initial = INITIAL
    full = FULL
    wire = WIRE
    for name, objective in [('full', full), ('wire', wire)]:
        value, grad = jax.jit(jax.value_and_grad(objective))(initial)
        g = np.asarray(grad)
        print('GRAD', name, float(value), float(g.min()), float(g.max()), float(np.linalg.norm(g)), int((g<0).sum()), flush=True)
        np.save('/output/grad-'+name+'.npy',g)
    # Independently accumulate weighted HPWL derivatives with NumPy.
    physical = np.asarray(initial) * SCALE
    all_centers = np.concatenate([physical, np.asarray(problem['port_positions'])],axis=0)
    owner = np.asarray(problem['pin_owner'],dtype=np.int64)
    pins = all_centers[owner]+np.asarray(problem['pin_offset'])
    starts=np.asarray(problem['net_offsets'],dtype=np.int64)[:-1]
    ids=np.repeat(np.arange(len(starts)),np.diff(problem['net_offsets']))
    hi=np.maximum.reduceat(pins,starts,axis=0);lo=np.minimum.reduceat(pins,starts,axis=0)
    maxmask=pins==hi[ids];minmask=pins==lo[ids]
    maxcounts=np.add.reduceat(maxmask.astype(float),starts,axis=0)
    mincounts=np.add.reduceat(minmask.astype(float),starts,axis=0)
    gp=(maxmask/maxcounts[ids]-minmask/mincounts[ids])*np.asarray(problem['net_weights'])[ids,None]/float(problem['wirelength_normalizer'])
    expected=np.zeros_like(all_centers,dtype=float);np.add.at(expected,owner,gp);expected=expected[:len(initial)]*SCALE
    print('REFERENCE',float(expected.min()),float(expected.max()),float(np.max(np.abs(g-expected))),flush=True)
    np.save('/output/expected-wire.npy',expected)
    return {'positions': np.array(problem['initial_positions'],copy=True)}
'''
 if model=='qwen':extra=extra.replace('INITIAL','initial_x_jax').replace('FULL','lambda z: compute_cost(z/canvas_jax)[0]').replace('WIRE','lambda z: probe_wire(z/canvas_jax)').replace('SCALE','1.0')
 else:extra=extra.replace('INITIAL','x_norm').replace('FULL','cost_fn').replace('WIRE','probe_wire').replace('SCALE','float(norm_factor)')
 fn.body+=ast.parse('def probe():\n'+extra).body[0].body
 s=ast.unparse(ast.fix_missing_locations(tree))+'\n';(p/(model+'-gradient.py')).write_text(s)
 old,new=('pin_centers = all_centers[owners_jax] + offsets_jax','pin_centers = jax.lax.optimization_barrier(all_centers[owners_jax] + offsets_jax)') if model=='qwen' else ('pins = centers[owners] + offsets','pins = jax.lax.optimization_barrier(centers[owners] + offsets)')
 assert s.count(old)==2
 (p/(model+'-gradient-barrier.py')).write_text(s.replace(old,new))
