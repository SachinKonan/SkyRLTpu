"""Every tpu.science module the shipped grading files import at module or call time must ship with them.

2026-11-02: cgroup_limits.envelope imports tpu.science.unit (added 2026-10-15), but unit.py was not in
GRADING_FILES, so every local math grade on a remote-only trainer died with ModuleNotFoundError and scored 0.
"""
import re
from pathlib import Path

from tpu.swarm.ray_train.overlay import GRADING_FILES

ROOT = Path(__file__).resolve().parents[2]
# Imported only on placement/routing paths that ship with SCIENCE_FILES, never on the math path.
SCIENCE_ONLY = {'placement_ray', 'ray_cpu', 'routing', 'routing_parallel', 'portfolio', 'portfolio_v2_grader',
                'challenge_contract'}


def science_imports(path):
    names = set()
    for line in (ROOT / path).read_text().splitlines():
        names.update(re.findall(r'^\s*from \.(\w+) import', line))
        names.update(re.findall(r'^\s*from tpu\.science\.(\w+) import', line))
        for group in re.findall(r'^\s*from \. import ([\w, ]+)', line):
            names.update(part.strip().split(' as ')[0] for part in group.split(','))
    return names


def test_grading_files_ship_their_science_imports():
    shipped = {p for p in GRADING_FILES if p.startswith('tpu/science/')}
    missing = {}
    for path in sorted(shipped):
        for name in science_imports(path) - SCIENCE_ONLY:
            target = f'tpu/science/{name}.py'
            if (ROOT / target).exists() and target not in shipped:
                missing.setdefault(target, []).append(path)
    assert not missing, f'imported by shipped grading files but not shipped: {missing}'


def test_grading_bundle_files_ship_their_science_imports():
    """build.GRADING_BUNDLE_FILES is the copy graders actually import (SCIENCE_WORKER_ROOT=$code at the archive
    root); it missed unit.py separately from GRADING_FILES, so the overlay fix alone did not reach the graders."""
    from tpu.swarm.ray_train.build import GRADING_BUNDLE_FILES
    shipped = {f'tpu/science/{name}' for name in GRADING_BUNDLE_FILES}
    missing = {}
    for path in sorted(shipped):
        for name in science_imports(path) - SCIENCE_ONLY:
            target = f'tpu/science/{name}.py'
            if (ROOT / target).exists() and target not in shipped:
                missing.setdefault(target, []).append(path)
    assert not missing, f'imported by GRADING_BUNDLE_FILES but not shipped: {missing}'
