import json
from pathlib import Path

import pytest

from tpu.science.placement_warm_start import CASES, DESTINATION, digest, verified_inputs


@pytest.fixture
def published(tmp_path):
    folder = tmp_path / DESTINATION
    folder.mkdir(parents=True)
    manifest = {'schema': 'xplace-start-v1', 'cases': {}}
    for case in CASES:
        problem = folder / f'{case}-problem.npz'
        problem.write_bytes(b'published-layout')
        native = tmp_path/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case
        native.mkdir(parents=True)
        for name in ['netlist.pb.txt', 'initial.plc']:
            (native/name).write_bytes(b'pinned-input')
        manifest['cases'][case] = {'valid': True, 'problem_sha256': digest(problem),
                                  'native_sha256': {name:digest(native/name) for name in ['netlist.pb.txt', 'initial.plc']}}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_published_inputs_are_complete(published):
    assert set(verified_inputs(published)) == set(CASES)


def test_changed_layout_is_rejected(published):
    (published/DESTINATION/'ibm01-problem.npz').write_bytes(b'replaced-layout')
    with pytest.raises(ValueError, match='Invalid Xplace'):
        verified_inputs(published)


def test_stale_netlist_is_rejected(published):
    native = published/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt'
    native.write_bytes(b'other-circuit')
    with pytest.raises(ValueError, match='stale native input'):
        verified_inputs(published)


def test_missing_case_is_rejected(published):
    path = published/DESTINATION/'manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['cases']['ibm18']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Incomplete'):
        verified_inputs(published)
