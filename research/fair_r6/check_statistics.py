"""Numerical controls separate marginal distribution from spatial structure."""
import argparse
import json
from pathlib import Path
import numpy as np
from noise_statistics import statistics


def main(a):
    with np.load(a.pilot,allow_pickle=False) as f:support=f['support']
    rng=np.random.default_rng(84016);z=rng.normal(size=(32,*support.shape))*support
    white=statistics(z,support)
    structured=z.copy();structured[:,support]=np.sort(z[:,support],axis=1)
    # The exact same one-point values are rearranged into a spatial trend.
    ordered=statistics(structured,support)
    for key in ('energy','ad','ad_shape','mean','variance'):
        assert np.allclose(white[key],ordered[key],rtol=1e-10,atol=1e-8)
    assert np.median(ordered['psd_cv'])>10*np.median(white['psd_cv'])
    assert np.median(ordered['correlation_energy'])>100*np.median(white['correlation_energy'])
    missing=z.copy();missing[:,~support]=np.nan
    missing_values=statistics(missing,support)
    assert all(np.array_equal(v,missing_values[k]) for k,v in white.items())
    reversed_values=statistics(z[::-1],support)
    singleton=statistics(z[:1],support)
    pointwise_errors={k:dict(reversal=float(np.max(abs(v-reversed_values[k][::-1]))),
                            singleton=float(np.max(abs(v[:1]-singleton[k])))) for k,v in white.items()}
    # Boolean gathers have different strides for one versus many rows. Float64
    # reductions can differ in their last bits; retain the measured errors.
    assert all(max(v.values())<1e-8 for v in pointwise_errors.values())
    shifted=statistics((2*z+.7)*support,support)
    for key in ('ad_shape','psd_cv','correlation_energy'):
        assert np.allclose(white[key],shifted[key],rtol=1e-9,atol=1e-8)
    zeros=statistics(np.zeros_like(z[:1]),support)
    expected=support.sum()*(2*np.log(2)-1)
    assert abs(zeros['ad'][0]-expected)<1e-8
    report=dict(kind='MASKED_GAUSSIAN_NUMERICAL_CONTROLS_NOT_COSMOLOGICAL_OOD',count=32,
                observed_dimensions=int(support.sum()),
                white_medians={k:float(np.median(v)) for k,v in white.items()},
                same_histogram_structured_medians={k:float(np.median(v)) for k,v in ordered.items()},
                missing_pixels_ignored=True,pointwise_absolute_errors=pointwise_errors,
                pointwise_tolerance=1e-8,
                spatial_statistics_respond_to_structure_without_changing_histogram=True,
                shape_statistics_invariant_to_per_map_affine_transform=True,
                zero_field_ad_analytic_error=float(abs(zeros['ad'][0]-expected)))
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
