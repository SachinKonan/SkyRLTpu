"""Frozen, independent per-map API for candidate 05.

This is an artifact-backed inference API, not an implementation of an as-yet
unspecified organizer private-evaluation interface. No fitting occurs here.
"""
import json
from pathlib import Path
import numpy as np
from common import sha
from detector import FrozenFlowDetector
from field_evaluate import predictor as field_predictor
from condition_field_scores import predict as regression_predict


class FrozenHybridDetector:
    def __init__(self, models, mask_file, summary_manifest_file, manifest_file,
                 regression_file):
        models = Path(models)
        self.manifest = json.loads(Path(manifest_file).read_text())
        m = self.manifest
        if m['weights'] != [.5, .5] or m['trace_probes'] != 1:
            raise ValueError('Unsupported frozen ensemble configuration')
        if sha(regression_file) != m['regression_sha256']:
            raise ValueError('Normal-data regression artifact changed')
        for model in m['field_models']:
            for name, digest in model['files'].items():
                if sha(models / model['name'] / name) != digest:
                    raise ValueError('Frozen field model changed')
        self.summary = FrozenFlowDetector(models, mask_file, summary_manifest_file)
        self.mask = self.summary.mask
        h, w = self.mask.shape
        self.count = self.mask.reshape(h//2, 2, w//2, 2).sum((1, 3))
        self.support = self.count >= 3
        self.fields = [field_predictor(models / model['name'],
                                      steps=m['ode_steps'],
                                      probe_seed=m['trace_seed'], batch=32)
                       for model in m['field_models']]
        with np.load(regression_file, allow_pickle=False) as f:
            self.mean = {k.removeprefix('mean_'): f[k]
                         for k in f.files if k.startswith('mean_')}
            self.variance = {k.removeprefix('variance_'): f[k]
                             for k in f.files if k.startswith('variance_')}
        self.mean['kind'] = str(self.mean['kind'])
        self.variance['kind'] = str(self.variance['kind'])

    def predict(self, observed):
        """Return scores in input order; larger means more anomalous.

        Accepts a valid-pixel vector, a full image, or batches of either.
        Missing survey pixels are ignored. Observed pixels must be finite.
        """
        x = np.asarray(observed)
        if x.shape == self.mask.shape or x.ndim == 1:
            x = x[None]
        if x.ndim == 2 and x.shape[1] == int(self.mask.sum()):
            full = np.zeros((len(x), *self.mask.shape), np.float32)
            full[:, self.mask] = x
        elif x.ndim == 3 and x.shape[1:] == self.mask.shape:
            full = np.where(self.mask, np.asarray(x, np.float32), 0.)
        else:
            raise ValueError('Expected masked images or valid-pixel vectors')
        if not len(full):
            return np.empty(0, np.float64)
        if not np.isfinite(full[:, self.mask]).all():
            raise ValueError('Nonfinite observed pixels')
        results = []
        h, w = self.mask.shape
        for lo in range(0, len(full), 16):
            batch = full[lo:lo+16]
            features = self.summary.features(batch)
            summary = np.mean([model(features)['minimum_base_energy']
                               for model in self.summary.models], axis=0).astype(np.float64)
            pooled = (batch.reshape(len(batch), h//2, 2, w//2, 2).sum((2, 4))
                      / np.maximum(self.count, 1) * self.support).astype(np.float32)
            nll = np.mean([np.asarray(model(pooled)['nll'], np.float64)
                           for model in self.fields], axis=0)
            sd = np.maximum(np.exp(np.clip(regression_predict(features, self.variance),
                                           -30, 30)/2),
                            self.manifest['conditional_std_floor'])
            residual = np.abs((nll - regression_predict(features, self.mean))/sd)
            sm, ss = self.manifest['summary_normal_center_scale']
            fm, fs = self.manifest['field_normal_center_scale']
            results.append(.5*((summary-sm)/ss + (residual-fm)/fs))
        result = np.concatenate(results)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite detector score')
        return result
