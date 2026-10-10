"""Weight check for the SS member (LCO) + full-fit synthetic view of v3-04/BIG_RE2 + SS."""
import sys
from pathlib import Path
import numpy as np
sys.argv = sys.argv[:1]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import lco_dif2 as M2  # builds LCO components (prints suppressed)
V304 = {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1}
BIG = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2}
for base, W in (('v3-04', V304), ('BIG_RE2', BIG)):
    for w in (1, 1.5, 2):
        M2.report(f'{base} + {w}SS', {**W, 'SS': w})
    M2.report(f'{base} + 1SS + 0.5nlo', {**W, 'SS': 1, 'nlo': .5})
with contextlib.redirect_stdout(io.StringIO()):
    import nlevel_eval as E
for base, W in (('v3-04', V304), ('BIG_RE2', BIG)):
    E.ev(f'[full-fit synth] {base}', E.mix(W))
    for w in (.5, 1):
        E.ev(f'[full-fit synth] {base} + {w}SS', E.mix({**W, 'SS': w}))
