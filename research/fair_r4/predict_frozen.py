"""Portable CLI for the frozen R4 detector bundle; no fitting or calibration."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main(a):
    sys.path.insert(0, str(Path(a.bundle).resolve()/'code/fair_r4'))
    from detector import FrozenR4Detector
    model = FrozenR4Detector(a.bundle)
    data = np.load(a.input, mmap_mode='r', allow_pickle=False)
    scores = model.predict(data, variant=a.variant)
    Path(a.output).write_text(json.dumps({'ood_scores': scores.tolist()}, allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','input','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--variant',choices=('08','09'),default='08');main(p.parse_args())
