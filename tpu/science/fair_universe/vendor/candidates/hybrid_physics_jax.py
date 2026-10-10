"""FAIR Phase-2 v2 hybrid starter: JAX multiscale features + JAX learned
summary encoder + InD-calibrated spectral/manifold/neighbor/NN detectors.

This is a research STARTER, not a proven SOTA model. Never sees synthetic
OoD labels during fit; never reads the reward split. No hidden-test access.

Public contract: fit(train, val, config, out_dir), load_model(...),
predict(model, noisy_maps, survey_mask, config) -> float[B].
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


@lru_cache(maxsize=6)
def make_extractor(h, w):
    # Full-mask FFTs are calibrated on identically masked InD examples.
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    min_freq = max(1.0 / max(h, w), .002)
    edges = np.geomspace(min_freq, .72, 17)
    radial_bins = jnp.asarray(np.clip(np.digitize(rad.ravel(), edges) - 1, 0, 15), jnp.int32)
    rcounts = jnp.maximum(jnp.bincount(radial_bins, length=16), 1)
    angle = np.arctan2(np.abs(fy), np.abs(fx) + 1e-6)
    angle = np.broadcast_to(angle, rad.shape)
    abins = jnp.asarray(np.minimum((4 * angle.ravel() / (np.pi / 2)).astype(int), 3), jnp.int32)
    acounts = jnp.maximum(jnp.bincount(abins, length=4), 1)
    kernel = jnp.asarray(np.stack([
        np.exp(-2 * np.pi ** 2 * s ** 2 * rad ** 2)
        for s in (1., 2.5, 5.5, 11.)
    ], axis=0), jnp.float32)
    # Four spatial stripes to expose spatial nonstationarity; stripe masks
    # have predictable geometry rather than dataset-specific indices.
    stripe_mask = jnp.asarray(np.stack([
        (np.arange(h)[:, None] >= q * h / 4) &
        (np.arange(h)[:, None] < (q + 1) * h / 4)
        for q in range(4)
    ]), jnp.float32)

    @jax.jit
    def extract(maps, mask):
        m = jnp.asarray(mask, jnp.float32)
        x = jnp.asarray(maps, jnp.float32) * m
        cnt = jnp.maximum(m.sum(), 1.)
        fft = jnp.fft.rfft2(x)
        p = jnp.abs(fft) ** 2 / cnt
        def radial(q):
            return jnp.log(jnp.bincount(radial_bins, weights=q.reshape(-1), length=16) /
                           rcounts + 1.e-8)
        spec = jax.vmap(radial)(p)
        def oriented(q):
            return jnp.log(jnp.bincount(abins, weights=q.reshape(-1), length=4) /
                           acounts + 1.e-8)
        angular = jax.vmap(oriented)(p)
        smask = jnp.fft.irfft2(jnp.fft.rfft2(m)[None] * kernel, s=(h, w))
        filtered = jnp.fft.irfft2(fft[:, None] * kernel[None], s=(h, w))
        filtered = filtered / jnp.maximum(smask[None], .1)
        fields = [x] + [filtered[:, i] for i in range(4)]
        # Finite-difference gradient statistics at each scale.
        valid_y = m * jnp.roll(m, 1, axis=0)
        valid_x = m * jnp.roll(m, 1, axis=1)
        # Remove wrap-around (invalid periodic image edges).
        valid_y = valid_y.at[0, :].set(0.)
        valid_x = valid_x.at[:, 0].set(0.)
        feats = [spec, angular]
        standardized = []
        for field in fields:
            mu = jnp.sum(field*m, axis=(1,2)) / cnt
            residual = (field - mu[:,None,None]) * m
            sigma = jnp.sqrt(jnp.sum(residual**2, axis=(1,2))/cnt + 1.e-9)
            z = residual / sigma[:, None, None]
            standardized.append(z)
            dz_y = (field - jnp.roll(field, 1, axis=1)) * valid_x
            dz_x = (field - jnp.roll(field, 1, axis=0)) * valid_y
            energy_y = (dz_y**2).sum((1,2)) / jnp.maximum(valid_x.sum(),1.)
            energy_x = (dz_x**2).sum((1,2)) / jnp.maximum(valid_y.sum(),1.)
            extras = jnp.stack([
                mu, jnp.log(sigma),
                jnp.sum(z**3*m, (1,2))/cnt,
                jnp.sum(z**4*m, (1,2))/cnt,
                jnp.sum((z > 2.)*m, (1,2))/cnt,
                jnp.sum((z < -2.)*m, (1,2))/cnt,
                jnp.sum((z > 3.)*m, (1,2))/cnt,
                jnp.log(energy_x + 1.e-8), jnp.log(energy_y + 1.e-8)
            ], axis=1)
            feats.append(extras)
        # Explicit wavelet-like bandpasses. Sensitive to non-Gaussianity
        # even when the image has approximately the same power spectrum.
        bands = [fields[i] - fields[i+1] for i in range(4)]
        for band in bands:
            mu = (band * m).sum((1,2)) / cnt
            z = (band - mu[:,None,None]) * m
            sd = jnp.sqrt((z**2).sum((1,2))/cnt + 1.e-9)
            z = z / sd[:,None,None]
            feats.append(jnp.stack([
                jnp.log(sd), (z**3*m).sum((1,2))/cnt,
                (z**4*m).sum((1,2))/cnt,
                (jnp.abs(z) > 2.5).astype(jnp.float32).sum((1,2))/cnt
            ], axis=1))
        for i, j in [(0,1),(1,2),(2,3),(3,4),(0,3)]:
            a, b = standardized[i], standardized[j]
            feats.append(jnp.stack([
                (a*b*m).sum((1,2))/cnt,
                (a*a*b*m).sum((1,2))/cnt,
                (a*a*b*b*m).sum((1,2))/cnt,
            ], axis=1))
        # Spatial heterogeneity at fixed slices, without relying on fixed IDs.
        stripes = stripe_mask * m[None]
        nstripe = jnp.maximum(stripes.sum((1,2)),1.)
        stripe_means = (x[:,None] * stripes[None]).sum((2,3)) / nstripe
        stripe_rms = jnp.sqrt((x[:,None]**2 * stripes[None]).sum((2,3))/nstripe + 1.e-9)
        feats += [stripe_means, jnp.log(stripe_rms)]
        return jnp.concatenate(feats, axis=1)
    return extract


def features(maps, mask):
    if maps.ndim != 3 or maps.shape[1:] != mask.shape:
        raise ValueError('maps[B,H,W] and mask[H,W] shapes required')
    return np.asarray(make_extractor(*mask.shape)(jnp.asarray(maps),
                    jnp.asarray(mask, jnp.float32)), dtype=np.float32)


def read_features(view, config, limit, seed):
    f, y = [], []
    for b in view.iter_batches(batch_size=int(config['batch_size']), epochs=1,
                               shuffle=True, seed=seed, limit=limit):
        f.append(features(b['maps'], b['mask']))
        y.append(np.asarray(b['labels'], np.float32))
    if not f:
        raise ValueError('No samples collected')
    return np.concatenate(f), np.concatenate(y)


def _init_mlp(rng, d, hidden, latent=32, out_y=5):
    p = {}
    for name, a, b in [('1',d,hidden),('2',hidden,latent),
                        ('rec',latent,d),('y',latent,out_y)]:
        rng, key = jax.random.split(rng)
        p['w'+name] = jax.random.normal(key,(a,b))*jnp.sqrt(2./(a+b))
        p['b'+name] = jnp.zeros((b,))
    return p


def _mlp(p, x):
    h = jax.nn.gelu(x @ p['w1'] + p['b1'])
    z = jax.nn.gelu(h @ p['w2'] + p['b2'])
    return z, z @ p['wrec'] + p['brec'], z @ p['wy'] + p['by']


def fit_neural(x, targets, seed, steps, batch, hidden):
    xj = jnp.asarray(np.clip(x,-8.,8.),jnp.float32)
    yj = jnp.asarray(targets,jnp.float32)
    rng = jax.random.PRNGKey(seed)
    p = _init_mlp(rng, x.shape[1], hidden)
    mom = jax.tree_util.tree_map(jnp.zeros_like,p)
    var = jax.tree_util.tree_map(jnp.zeros_like,p)

    @jax.jit
    def one_step(p,mom,var,xb,yb,key,step):
        def obj(q):
            mask = jax.random.bernoulli(key, .88, shape=xb.shape)
            zin = xb * mask
            z, rec, yy = _mlp(q, zin)
            _, _, yy_full = _mlp(q, xb)
            loss_y = jnp.mean((yy-yb)**2)
            loss_r = jnp.mean((rec-xb)**2)
            loss_c = jnp.mean((yy_full-yy)**2)
            return loss_y + .12*loss_r + .10*loss_c
        loss, grads = jax.value_and_grad(obj)(p)
        mom = jax.tree_util.tree_map(lambda m,g:.9*m+.1*g,mom,grads)
        var = jax.tree_util.tree_map(lambda v,g:.999*v+.001*g*g,var,grads)
        t = step+1
        rate = .0012 * (1. - .8*step/jnp.maximum(steps,1))
        p = jax.tree_util.tree_map(lambda w,m,v: w - rate*(m/(1-.9**t)) /
                                   (jnp.sqrt(v/(1-.999**t))+1e-6),p,mom,var)
        return p,mom,var,loss

    rng_np = np.random.default_rng(seed)
    losses = []
    for step in range(int(steps)):
        ix = rng_np.integers(len(x),size=min(batch,len(x)))
        rng, key = jax.random.split(rng)
        p,mom,var,loss = one_step(p,mom,var,xj[ix],yj[ix],key,jnp.asarray(step))
        if step % 50 == 0 or step == int(steps)-1:
            losses.append(float(loss))
    return {key: np.asarray(v) for key,v in p.items()}, losses


@jax.jit
def _neural_infer_jit(params, x):
    return _mlp(params, x)


def _nn_outputs(p, x):
    params = {k:jnp.asarray(v,jnp.float32) for k,v in p.items()}
    zs, recs, ys = [],[],[]
    for i in range(0,len(x),512):
        z,r,y = _neural_infer_jit(params, jnp.asarray(np.clip(x[i:i+512],-8.,8.),jnp.float32))
        zs.append(np.asarray(z)); recs.append(np.asarray(r)); ys.append(np.asarray(y))
    return np.concatenate(zs),np.concatenate(recs),np.concatenate(ys)


def _whitener(x, ridge=.1):
    cov = LedoitWolf().fit(x).covariance_
    lam, vec = np.linalg.eigh(cov + float(ridge)*np.eye(cov.shape[0]))
    return (vec/np.sqrt(np.maximum(lam,1.e-7)))@vec.T


def _distance_to_refs(query, refs, k=5):
    # Batch to avoid O(N*reference_count) resident memory.
    out = np.empty((len(query),), np.float64)
    ref2 = np.sum(refs*refs,axis=1)[None]
    for i in range(0,len(query),256):
        q = query[i:i+256]
        sq = np.maximum(np.sum(q*q,axis=1)[:,None]+ref2-2*q@refs.T,0.)
        n = min(k, sq.shape[1])
        out[i:i+256] = np.partition(sq,n-1,axis=1)[:,:n].mean(axis=1)
    return out


def raw_scores(model, feat):
    xx = np.clip((feat - model['center'])/model['scale'],-8.,8.)
    zz = (xx - model['pca_mean']) @ model['pca_components'].T
    sp = xx[:, :16] - model['spec_mean']
    spectral = np.einsum('ni,ij,nj->n',sp,model['spec_precision'],sp)
    glob = ((zz-model['z_mean']) @ model['global_whiten'])
    global_score = (glob*glob).sum(axis=1)
    cond = (zz @ model['conditional_whiten'])
    centers = model['cosmo_centers'] @ model['conditional_whiten']
    d2 = np.maximum((cond*cond).sum(1)[:,None]+(centers*centers).sum(1)[None]-2*cond@centers.T,0.)
    conditional = d2.min(axis=1)
    knn = _distance_to_refs(glob,model['references'],k=5)
    _,recons,labels = _nn_outputs({key[3:]:v for key,v in model.items() if key.startswith('nn_')}, xx)
    nn_residual = np.mean((xx - recons)**2, axis=1)
    # Physics head uncertainty is not used as a direct anomaly score: it may
    # confidently predict wrong cosmology on shifted data.
    return np.column_stack((spectral,global_score,conditional,knn,nn_residual))


def fit(train, val, config, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    bs = int(config['batch_size'])
    steps = int(config['steps'])
    if bs<1 or steps<1:
        raise ValueError('steps, batch_size must be positive')
    n = min(len(train), max(bs,steps*bs))
    f,labels = read_features(train,config,n,int(config['seed']))
    center = np.median(f,axis=0)
    scale = np.maximum(np.std(f,axis=0),1.e-4)
    x = np.clip((f-center)/scale,-8.,8.)
    dims = min(32,x.shape[1],len(x)-1)
    if dims < 2:
        raise ValueError('Need at least three training examples')
    pca = PCA(n_components=dims,svd_solver='full').fit(x)
    z = pca.transform(x)
    z_mean = z.mean(0)
    global_whiten = _whitener(z,.15)
    # Same cosmology, different nuisances -> tied-shrinkage conditional density.
    groups,inverse = np.unique(np.round(labels[:, :2],5),axis=0,return_inverse=True)
    if len(groups)>500 or np.bincount(inverse).min()<3:
        from sklearn.cluster import KMeans
        inverse = KMeans(n_clusters=min(24,max(2,len(z)//16)),n_init=3,
                         random_state=int(config['seed'])).fit_predict(z)
    cosmo_centers = np.stack([z[inverse==j].mean(0) for j in range(inverse.max()+1)])
    conditional_whiten = _whitener(z-cosmo_centers[inverse],.15)
    rng = np.random.default_rng(int(config['seed']))
    rid = rng.choice(len(z),size=min(1500,len(z)),replace=False)
    references = ((z-z_mean) @ global_whiten)[rid]
    spec_x = x[:,:16]
    spec_cov = LedoitWolf().fit(spec_x)
    spec_mean = spec_cov.location_
    spec_precision = spec_cov.precision_
    ys = np.mean(labels,axis=0)
    yscale = np.maximum(np.std(labels,axis=0),.01)
    p, losses = fit_neural(x,(labels-ys)/yscale,seed=int(config['seed']),
                           steps=steps,batch=bs,
                           hidden=int(config.get('hybrid_hidden',96)))
    model = dict(center=center.astype(np.float64),scale=scale.astype(np.float64),
                 pca_mean=pca.mean_.astype(np.float64),
                 pca_components=pca.components_.astype(np.float64),
                 z_mean=z_mean.astype(np.float64),
                 global_whiten=global_whiten,cosmo_centers=cosmo_centers,
                 conditional_whiten=conditional_whiten,references=references,
                 spec_mean=spec_mean,spec_precision=spec_precision,
                 label_center=ys,label_scale=yscale,
                 **{'nn_'+k:np.asarray(v) for k,v in p.items()})
    # Calibrate on fresh noisy InD realizations, distinct from the fitted data.
    v,_ = read_features(val,config,min(len(val),int(config.get('calibration_max',2048))),
                        int(config['seed'])+918)
    rs = raw_scores(model,v)
    model['calibration'] = np.sort(rs,axis=0)
    np.savez_compressed(out/'model.npz',**model)
    return {'train_examples':len(f),'calibration_examples':len(v),
            'num_features':int(f.shape[1]),'pca_dim':int(dims),
            'cosmology_centers':len(cosmo_centers),
            'jax_neural_train_steps':steps,'neural_loss_trace':losses}


def load_model(out_dir,config):
    with np.load(Path(out_dir)/'model.npz') as f:
        return {k:f[k] for k in f.files}


def all_scores(model, maps, mask, config):
    rs = raw_scores(model,features(maps,mask))
    cal = model['calibration']
    n = len(cal)
    # Convert diverse detector units to comparable InD tail surprise.
    tail = np.stack([-np.log((n+1-np.searchsorted(cal[:,i],rs[:,i],side='right'))/
                                (n+1.)) for i in range(rs.shape[1])],axis=1)
    # The two complementary ensembles discourage reliance on a single detector.
    weighted = tail @ np.asarray([.35,.05,.25,.20,.15])
    robust = .60*tail[:,0] + .25*tail[:,2] + .15*tail[:,3]
    top2 = np.sort(tail,axis=1)[:,-2:].mean(axis=1)
    return {
       'spectral':tail[:,0], 'global':tail[:,1],
       'conditional':tail[:,2], 'neighbor':tail[:,3], 'neural':tail[:,4],
       'hybrid':weighted, 'hybrid_robust':robust, 'hybrid_top2':top2,
       'hybrid_max':tail.max(axis=1),
    }


def predict(model,maps,mask,config):
    mode = str(config.get('hybrid_mode','hybrid_robust'))
    choices = all_scores(model,maps,mask,config)
    if mode not in choices:
        raise ValueError(f'Invalid hybrid_mode {mode}; choose from {list(choices)}')
    return np.asarray(choices[mode],dtype=np.float64)
