"""Numerical change-of-variables and per-sample independence checks."""
import json
import jax
import jax.numpy as j
import numpy as np
from flow import AffineFlow

net=AffineFlow(6,3,16);key=jax.random.PRNGKey(7);x=jax.random.normal(key,(4,6));c=jax.random.normal(jax.random.PRNGKey(8),(4,2))
params=net.init(key,x,c)['params']
keys=jax.random.split(jax.random.PRNGKey(10),len(jax.tree.leaves(params)))
params=jax.tree.unflatten(jax.tree.structure(params),[v+.05*jax.random.normal(k,v.shape) for v,k in zip(jax.tree.leaves(params),keys)])
# Compare an autodifferentiated full Jacobian with the analytical coupling sum.
def latent(one,condition):
    return net.apply({'params':params},one[None],condition[None],return_latent=True)[0]
n,e,ld=net.apply({'params':params},x,c)
jac=jax.jacfwd(latent,argnums=0)(x[0],c[0]);actual=j.linalg.slogdet(jac)[1]
assert np.allclose(actual,ld[0],rtol=2e-5,atol=2e-5),(actual,ld[0])
z=latent(x[0],c[0]);analytic=.5*(j.square(z).sum()+6*j.log(2*j.pi))-actual
assert np.allclose(analytic,n[0],rtol=2e-5,atol=2e-5)
alone=net.apply({'params':params},x[:1],c[:1])[0]
reverse=net.apply({'params':params},x[::-1],c[::-1])[0][::-1]
assert np.allclose(n[:1],alone,rtol=1e-5,atol=1e-5) and np.allclose(n,reverse,rtol=1e-5,atol=1e-5)
grad=jax.grad(lambda p:net.apply({'params':p},x,c)[0].mean())(params)
assert all(np.isfinite(v).all() for v in jax.tree.leaves(grad))
print(json.dumps({'jacobian_logdet':float(actual),'reported_logdet':float(ld[0]),'analytic_nll':float(analytic),'reported_nll':float(n[0]),'finite_gradient':True,'batch_independence':True}),flush=True)
