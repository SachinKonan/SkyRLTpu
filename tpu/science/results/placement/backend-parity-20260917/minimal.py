import numpy as np
import jax
import jax.numpy as jnp

def place(problem, seed, *, time_budget_s):
    x=jnp.array([[1.,9.],[3.,4.],[2.,5.],[8.,2.],[4.,7.],[6.,3.]],dtype=jnp.float32)
    ids=jnp.array([0,0,1,1,2,2],dtype=jnp.int32)
    for sorted_ids in [False,True]:
        for op in ['max','min','range','sum']:
            def f(z):
                hi=jax.ops.segment_max(z,ids,3,indices_are_sorted=sorted_ids)
                lo=jax.ops.segment_min(z,ids,3,indices_are_sorted=sorted_ids)
                y={'max':hi,'min':lo,'range':hi-lo,'sum':jax.ops.segment_sum(z,ids,3)}[op]
                return .001*jnp.sum(y)
            value,grad=jax.jit(jax.value_and_grad(f))(x)
            print('MINIMAL',sorted_ids,op,float(value),np.asarray(grad).tolist(),flush=True)
    def dense(z):
        pairs=z.reshape(3,2,2)
        return .001*jnp.sum(jnp.max(pairs,axis=1)-jnp.min(pairs,axis=1))
    value,grad=jax.jit(jax.value_and_grad(dense))(x)
    print('DENSE',float(value),np.asarray(grad).tolist(),flush=True)
    return {'positions':np.array(problem['initial_positions'],copy=True)}
