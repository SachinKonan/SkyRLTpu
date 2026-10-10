"""Independent geometry checks against connected components and holes."""
import json
import numpy as np
import jax
from scipy.ndimage import label,generate_binary_structure
from spatial_features import digital_geometry,SpatialSummary

patterns=[]
single=np.zeros((16,16),bool);single[5,5]=1;patterns.append(single)
diagonal=single.copy();diagonal[6,6]=1;patterns.append(diagonal)
block=np.zeros((16,16),bool);block[3:9,3:9]=1;patterns.append(block)
ring=block.copy();ring[4:8,4:8]=0;patterns.append(ring)
patterns += [np.zeros((16,16),bool),np.ones((16,16),bool)]
rng=np.random.default_rng(6201)
patterns += [rng.random((16,16))<p for p in (.1,.3,.5,.7,.9) for _ in range(5)]
binary=np.asarray(patterns);actual=np.asarray(jax.jit(digital_geometry)(binary))
for b,v in zip(binary,actual):
    p=np.pad(b,1)
    components=label(p,generate_binary_structure(2,1))[1]
    holes=label(~p,generate_binary_structure(2,2))[1]-1
    perimeter=np.count_nonzero(p[1:]!=p[:-1])+np.count_nonzero(p[:,1:]!=p[:,:-1])
    assert np.array_equal(v,np.asarray([b.sum(),perimeter,components-holes])),(v,components,holes)
mask=np.ones((64,48),bool);mask[8:11,8:11]=0
fn=SpatialSummary(mask,batch=4);x=rng.normal(size=(3,64,48)).astype(np.float32)
values=fn(x)
assert values.shape==(3,93) and np.isfinite(values).all()
assert np.allclose(values[:1],fn(x[:1]),rtol=1e-6,atol=1e-6)
assert np.allclose(values,fn(x[::-1])[::-1],rtol=1e-6,atol=1e-6)
missing=x.copy();missing[:,~mask]=np.nan
assert np.array_equal(values,fn(missing))
assert fn(np.empty((0,64,48))).shape==(0,93)
print(json.dumps(dict(geometry_patterns=len(patterns),euler_reference='4-connected components minus 8-connected holes',
                      finite=True,pointwise=True,unobserved_nan_ignored=True,description=fn.description)),flush=True)
