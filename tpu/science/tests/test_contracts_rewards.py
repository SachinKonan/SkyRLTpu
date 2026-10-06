import math
import numpy as np
import pytest
from tpu.science.contracts import rust_literal, weights, placement as legal
from tpu.science.rewards import qubit, placement, portfolio
from tpu.science.portfolio_replay import replay, transition


def test_literal_never_executes():
    assert rust_literal('RUST_CODE = r"fn x() {}"') == 'fn x() {}'
    for bad in ['RUST_CODE = str(1)', 'RUST_CODE = "a"; print(1)', 'x = RUST_CODE = "a"', 'RUST_CODE: str = "a"']:
        with pytest.raises(ValueError):rust_literal(bad)


def test_cost_rankings():
    assert qubit([30],[30],[1])[0] == .5
    assert qubit([30],[60],[1])[0] < .5 < qubit([30],[0],[1])[0]
    assert placement([[1,2,3]],[[1,2,3]])[0] == .5
    with pytest.raises(ValueError):placement([[1,0,3]],[[1,2,3]])
    with pytest.raises(ValueError):qubit([0],[0],[1])


def test_sharpe_sample_and_negative_valid():
    r=np.array([-.02,.01,-.03,.005])
    reward,m=portfolio(r,np.zeros(4),np.zeros(4))
    assert m['sharpe'] == pytest.approx(math.sqrt(252)*r.mean()/r.std(ddof=1))
    assert 0 < reward < .5
    assert portfolio(np.zeros(4),np.zeros(4),np.zeros(4))[0] == .5


def test_action_constraints():
    for bad in [[1], [0,0], [-.1,1.1], [float('nan'),1]]:
        with pytest.raises(ValueError):weights(bad,1)
    np.testing.assert_allclose(weights([0,1],1),[0,1])


def test_fees_and_drift():
    r,w,t=transition(np.array([1.,0.,0.]),[0,.5,.5],[.1,-.1],0,.001)
    assert r == pytest.approx(-.001)
    np.testing.assert_allclose(w,[0,.55,.45]);assert t == 1
    r,w,t=transition(w,w,[0,0],0,.001)
    assert r == 0 and t == 0


def test_replay_causal_windows():
    f=np.arange(5,dtype=float).reshape(5,1,1)
    data=dict(features=f,context_features=np.arange(-19,0).reshape(19,1,1),returns=np.ones((5,1))*.01,cash_returns=np.zeros(5))
    seen=[]
    def cash(obs):
        assert set(obs)=={'features','current_weights'}
        seen.append(obs['features'][-1,0,0]);return [1.,0.]
    r,m,_=replay(data,cash)
    assert seen==list(range(5)) and r==.5 and m['cumulative_return']==0


def test_rectangle_legality_not_cell_uniqueness():
    p=dict(width=10,height=10,num_cols=10,num_rows=10,movable_macros=[dict(width=3,height=3,allowed_orientations=['N'])]*2,
           fixed_objects=[],blockages=[])
    # Distinct adjacent centers still overlap full macro footprints.
    with pytest.raises(ValueError):legal(p,dict(cell_ids=[44,45],orientations=['N','N']))
    assert legal(p,dict(cell_ids=[22,77],orientations=['N','N']))['cell_ids']==[22,77]
