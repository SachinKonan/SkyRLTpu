from pathlib import Path
import pytest
from tpu.science.resources import PORTFOLIO_LIBRARIES, CVXPY_SOLVERS, library_environment, portfolio_prompt_resources
from tpu.science.portfolio import check_imports


def test_public_prompt_and_allowlist_match():
    prompt=portfolio_prompt_resources()
    for library in PORTFOLIO_LIBRARIES:
        assert library in prompt
        check_imports(f'import {library}')
    for solver in CVXPY_SOLVERS:assert solver in prompt
    for library in ['jax','torch','flax','optax','socket','subprocess']:
        with pytest.raises(ValueError):check_imports(f'import {library}')


def test_thread_settings_avoid_nested_oversubscription():
    env=library_environment(4)
    assert env['LOKY_MAX_CPU_COUNT']==env['OMP_THREAD_LIMIT']==env['NUMEXPR_MAX_THREADS']=='4'
    assert env['OPENBLAS_NUM_THREADS']==env['MKL_NUM_THREADS']==env['OMP_NUM_THREADS']=='1'
    assert env['MPLBACKEND']=='Agg'
    with pytest.raises(ValueError):library_environment(0)


def test_rendered_prompt_is_current():
    from tpu.science.prepare_prompts import build
    rendered=Path(__file__).parents[1]/'prompts/rendered/portfolio.txt'
    assert portfolio_prompt_resources() in rendered.read_text()


@pytest.mark.parametrize('solver', CVXPY_SOLVERS)
def test_installed_cpu_solvers(solver):
    import cvxpy as cp
    import numpy as np
    assert solver in cp.installed_solvers()
    x=cp.Variable(2)
    problem=cp.Problem(cp.Minimize(-x[0]),[x>=0,cp.sum(x)==1,x[0]<=.8])
    problem.solve(solver=solver)
    assert problem.status=='optimal'
    np.testing.assert_allclose(x.value,[.8,.2],atol=1e-4)
