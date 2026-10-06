import numpy as np
import jax
import jax.numpy as jnp

def place(problem, seed, *, time_budget_s):
    p=problem
    canvas=jnp.asarray(p['canvas'],dtype=jnp.float32)
    x=jnp.asarray(p['initial_positions'],dtype=jnp.float32)/canvas
    ports=jnp.asarray(p['port_positions'],dtype=jnp.float32)/canvas
    owners=jnp.asarray(p['pin_owner'],dtype=jnp.int32)
    offsets=jnp.asarray(p['pin_offset'],dtype=jnp.float32)/canvas
    w=jnp.asarray(p['net_weights'],dtype=jnp.float32)
    ids=jnp.asarray(np.repeat(np.arange(len(w)),np.diff(p['net_offsets'])),dtype=jnp.int32)
    norm=float(p['wirelength_normalizer'])
    pins=jnp.concatenate([x,ports])[owners]+offsets
    def fp(z):
        hi=jax.ops.segment_max(z,ids,len(w))
        lo=jax.ops.segment_min(z,ids,len(w))
        return jnp.sum(w*jnp.sum((hi-lo)*canvas,axis=1))/norm
    def fc(z):return fp(jax.lax.optimization_barrier(jnp.concatenate([z,ports])[owners]+offsets))
    grads={}
    for name,fn,arg in [('pins',fp,pins),('centers',fc,x),('gather',lambda z:.001*jnp.sum(jnp.concatenate([z,ports])[owners]),x)]:
        val,grad=jax.jit(jax.value_and_grad(fn))(arg)
        g=np.asarray(grad);grads[name]=g
        print('WIRE',name,float(val),g.shape,float(g.min()),float(g.max()),float(np.linalg.norm(g)),int((g<0).sum()),flush=True)
        np.save('/output/grad-'+name+'.npy',g)
    manual=np.zeros((len(x)+len(ports),2),dtype=np.float32)
    np.add.at(manual,np.asarray(owners),grads['pins'])
    print('CHAIN',float(np.max(np.abs(manual[:len(x)]-grads['centers']))),flush=True)
    for n in [4,128,1024]:
        vals=jnp.arange(n*2,dtype=jnp.float32).reshape(n,2)
        idx=jnp.asarray(np.arange(n*3)%n,dtype=jnp.int32)
        f=lambda z:.001*jnp.sum(z[idx])
        g=np.asarray(jax.jit(jax.grad(f))(vals))
        print('GATHER',n,float(g.min()),float(g.max()),flush=True)
    return {'positions':np.array(p['initial_positions'],copy=True)}
