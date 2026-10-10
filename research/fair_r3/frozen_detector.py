"""Frozen pointwise APIs for R3 candidates 06 and 07. No adaptation or fitting."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r2'))
sys.path.insert(2, str(Path(__file__).resolve().parents[1] / 'fair_ae'))
from common import sha
from flow import predictor as summary_predictor
from field_evaluate import predictor as field_predictor
from condition_field_scores import predict as background_predict
from features import make_predict
from spatial_features import SpatialSummary
from spatial_residuals import load as load_spatial, predict as spatial_predict
from calibrate_spatial import regress


class FrozenR3Detector:
    def __init__(self, bundle):
        root = Path(bundle); self.manifest = json.loads((root / 'manifest.json').read_text())
        m = self.manifest
        if m['format'] != 'FAIR_R3_POINTWISE_V1' or m['summary_weights'] != [1/3]*3:
            raise ValueError('Unsupported frozen detector definition')
        for path, digest in m['files'].items():
            if sha(root / path) != digest:
                raise ValueError('Frozen artifact changed: ' + path)
        self.mask = np.load(root / 'mask.npy', allow_pickle=False)
        self.physical = make_predict(self.mask, 16)
        self.spatial = SpatialSummary(self.mask, 16)
        self.summaries = [summary_predictor(root / 'models' / name) for name in m['summary_models']]
        self.fields = [field_predictor(root / 'models' / name, steps=m['ode_steps'],
                                      probe_seed=m['trace_seed'], batch=32) for name in m['field_models']]
        self.region_model = load_spatial(root / 'spatial-residual.npz')
        with np.load(root / 'spatial-centering.npz', allow_pickle=False) as f:
            self.region_mean = {k.removeprefix('mean_'): f[k] for k in f.files if k.startswith('mean_')}
        with np.load(root / 'field-regressors.npz', allow_pickle=False) as f:
            self.field_mean = {k.removeprefix('mean_'): f[k] for k in f.files if k.startswith('mean_')}
            self.field_variance = {k.removeprefix('variance_'): f[k] for k in f.files if k.startswith('variance_')}
        self.field_mean['kind'] = str(self.field_mean['kind'])
        self.field_variance['kind'] = str(self.field_variance['kind'])
        h, w = self.mask.shape
        self.count = self.mask.reshape(h//2, 2, w//2, 2).sum((1, 3))
        self.support = self.count >= 3

    def _full(self, observed):
        x = np.asarray(observed)
        if x.shape == self.mask.shape or x.ndim == 1: x = x[None]
        if x.ndim == 2 and x.shape[1] == int(self.mask.sum()):
            full = np.zeros((len(x), *self.mask.shape), np.float32); full[:, self.mask] = x
        elif x.ndim == 3 and x.shape[1:] == self.mask.shape:
            full = np.where(self.mask, np.asarray(x, np.float32), 0.)
        else:
            raise ValueError('Expected valid-pixel vectors or full survey maps')
        if not np.isfinite(full[:, self.mask]).all(): raise ValueError('Nonfinite observed pixels')
        return full

    def summary_spatial(self, full):
        f = self.physical(full).astype(np.float64)
        extended = np.column_stack([f, self.spatial(full)]).astype(np.float64)
        summary = np.mean([fn(f)['minimum_base_energy'] for fn in self.summaries], axis=0).astype(np.float64)
        raw = spatial_predict(extended, self.region_model)
        regional = np.log(np.maximum(raw, 1e-12)) - regress(f, self.region_mean)
        return dict(summary=summary, regional=regional, features=f)

    def combine(self, summary, field_score, regional):
        sm, ss = self.manifest['summary_normal_center_scale']
        fm, fs = self.manifest['field_normal_center_scale']
        hm, hs = self.manifest['hybrid_normal_center_scale']
        rm, rs = self.manifest['regional_normal_center_scale']
        hybrid = .5 * ((summary - sm) / ss + (field_score - fm) / fs)
        spatial_hybrid = np.maximum((hybrid - hm) / hs, (regional - rm) / rs)
        values = {'06': hybrid, '07': spatial_hybrid}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite detector score')
        return values

    def predict_both(self, observed):
        full = self._full(observed)
        if not len(full): return {k: np.empty(0, np.float64) for k in ('06', '07')}
        result = {k: [] for k in ('06', '07')}; h, w = self.mask.shape
        for lo in range(0, len(full), 16):
            batch = full[lo:lo+16]; comp = self.summary_spatial(batch)
            pooled = (batch.reshape(len(batch), h//2, 2, w//2, 2).sum((2, 4)) /
                      np.maximum(self.count, 1) * self.support).astype(np.float32)
            nll = np.mean([np.asarray(fn(pooled)['nll'], np.float64) for fn in self.fields], axis=0)
            sd = np.maximum(np.exp(np.clip(background_predict(comp['features'], self.field_variance), -30, 30)/2),
                            self.manifest['conditional_std_floor'])
            field = np.abs((nll - background_predict(comp['features'], self.field_mean)) / sd)
            for k, v in self.combine(comp['summary'], field, comp['regional']).items(): result[k].append(v)
        return {k: np.concatenate(v) for k, v in result.items()}

    def predict(self, observed, variant='07'):
        if variant not in ('06','07'): raise ValueError('Expected variant 06 or 07')
        return self.predict_both(observed)[variant]
