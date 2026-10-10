"""Frozen independent-map APIs: continuous physical profile and normal correction."""
import json
import sys
from pathlib import Path
import numpy as np

# R3's legacy helper imports include `compare`, a name also present in R2.
# Keep the R3 directory first while loading that unchanged runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'fair_r3'))
from frozen_detector import FrozenR3Detector
from spatial_residuals import predict as spatial_predict
from calibrate_spatial import regress
from condition_field_scores import predict as background_predict
from continuous_profile import Profile
from score_correction import load, score
from common import sha


class FrozenR4Detector:
    def __init__(self, bundle):
        root = Path(bundle); self.manifest = json.loads((root/'r4-manifest.json').read_text())
        if self.manifest['format'] != 'FAIR_R4_POINTWISE_V1':
            raise ValueError('Unsupported frozen detector definition')
        for path, digest in self.manifest['files'].items():
            if sha(root/path) != digest: raise ValueError('Frozen artifact changed: '+path)
        self.base = FrozenR3Detector(root/'base')
        self.profiles = [Profile(root/'base/models'/name) for name in self.base.manifest['summary_models']]
        self.correction = load(root/'normal-score-regressors.npz')

    def summary_components(self, features165):
        f = np.asarray(features165[:, :72], np.float64)
        results = [p(f) for p in self.profiles]
        grid = np.stack([v['grid_energy'] for v in results]).mean(0)
        continuous = np.stack([v['continuous_energy'] for v in results]).mean(0)
        coordinates = np.stack([v['continuous_theta'] for v in results]).mean(0)
        raw = spatial_predict(np.asarray(features165, np.float64), self.base.region_model)
        regional = np.log(np.maximum(raw, 1e-12))-regress(f, self.base.region_mean)
        corrected = score(continuous, coordinates, self.correction)
        return dict(grid=grid, continuous=continuous, coordinates=coordinates, regional=regional, **corrected)

    def combine(self, components, field_score):
        b = self.base; m = b.manifest
        output = b.combine(components['grid'], field_score, components['regional'])
        field = (field_score-m['field_normal_center_scale'][0])/m['field_normal_center_scale'][1]
        for mode in ('continuous', 'centered', 'scaled'):
            cm, cs = self.manifest['normal_center_scale'][mode]
            hybrid = .5*((components[mode]-cm)/cs+field)
            hm, hs = self.manifest['normal_center_scale'][mode+'-hybrid']
            rm, rs = m['regional_normal_center_scale']
            output[mode+'-hybrid'] = hybrid
            output[mode+'-spatial'] = np.maximum((hybrid-hm)/hs,(components['regional']-rm)/rs)
        output['08'] = output['scaled-hybrid']; output['09'] = output['scaled-spatial']
        if not all(np.isfinite(v).all() for v in output.values()):
            raise ValueError('Nonfinite detector score')
        return output

    def predict_both(self, observed):
        b = self.base; full = b._full(observed)
        if not len(full): return {k: np.empty(0, np.float64) for k in ('08', '09')}
        h, w = b.mask.shape; result = {k: [] for k in ('08', '09')}
        for lo in range(0, len(full), 16):
            batch = full[lo:lo+16]
            f = b.physical(batch).astype(np.float64)
            extended = np.column_stack([f, b.spatial(batch)])
            components = self.summary_components(extended)
            pooled = (batch.reshape(len(batch), h//2, 2, w//2, 2).sum((2, 4)) /
                      np.maximum(b.count, 1)*b.support).astype(np.float32)
            nll = np.mean([np.asarray(fn(pooled)['nll'], np.float64) for fn in b.fields], axis=0)
            sd = np.maximum(np.exp(np.clip(background_predict(f,b.field_variance),-30,30)/2),
                            b.manifest['conditional_std_floor'])
            field = abs((nll-background_predict(f,b.field_mean))/sd)
            combined = self.combine(components, field)
            for key in result: result[key].append(combined[key])
        return {k: np.concatenate(v) for k, v in result.items()}

    def predict(self, observed, variant='08'):
        if variant not in ('08', '09'): raise ValueError('Expected variant 08 or 09')
        return self.predict_both(observed)[variant]
