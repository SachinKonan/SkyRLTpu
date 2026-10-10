"""model.Likelihood with a configurable jitter mask (adds cosmology jitter on Omega_m/S8 inputs).

The fit method is model.Likelihood.fit verbatim except the jitter mask line.
"""
import inspect
import textwrap

import model
from model import Likelihood

_src = textwrap.dedent(inspect.getsource(Likelihood.fit))
_old = "self.jmask = torch.tensor([0., 0., 1., 1., 0.5], device=self.dev)"
assert _old in _src
_src = _src.replace(_old, "self.jmask = torch.tensor(self.jm, device=self.dev)")
_ns = {}
exec(_src, model.__dict__, _ns)


class JLikelihood(Likelihood):
    def __init__(self, dev='cpu', eig_floor=0.02, jm=(0., 0., 1., 1., 0.5)):
        super().__init__(dev, eig_floor)
        self.jm = [float(v) for v in jm]

    fit = _ns['fit']
