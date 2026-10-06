"""CPU integration pilot: price learner plus long-only minimum-variance weights."""
import numpy as np
import scipy.special
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import xgboost as xgb
import cvxpy as cp
import sympy
from matplotlib.figure import Figure


def fit(train, valid, rng, *, time_budget_s):
    frame=pd.DataFrame(train['features'].reshape(-1,train['features'].shape[-1]))
    scaler=StandardScaler().fit(frame)
    x=scaler.transform(frame)
    y=train['returns'].reshape(-1)
    ols=sm.OLS(y,sm.add_constant(x)).fit()
    booster=xgb.XGBRegressor(n_estimators=64,max_depth=4,n_jobs=4,device='cpu',
        tree_method='hist',random_state=int(rng.integers(2**31)),verbosity=0)
    booster.fit(x,y)
    covariance=np.cov(train['returns'],rowvar=False)+np.eye(train['returns'].shape[1])*1e-6
    w=cp.Variable(covariance.shape[0])
    problem=cp.Problem(cp.Minimize(cp.quad_form(w,cp.psd_wrap(covariance))),[w>=0,cp.sum(w)==1])
    problem.solve(solver='CLARABEL',max_iter=100,time_limit=10.,max_threads=1)
    if w.value is None:raise ValueError('minimum-variance solver failed')
    base=np.maximum(w.value,0);base/=base.sum()
    # Small in-memory smoke checks of the other allowed scientific packages.
    assert sympy.integrate(sympy.Symbol('x'),(sympy.Symbol('x'),0,1))==sympy.Rational(1,2)
    figure=Figure();figure.subplots().plot(base);figure.clear()
    return dict(scaler=scaler,booster=booster,ols=np.asarray(ols.params),base=base)


def act(model,obs):
    x=model['scaler'].transform(obs['features'][-1])
    prediction=.5*model['booster'].predict(x)+.5*(model['ols'][0]+x@model['ols'][1:])
    tilted=model['base']*scipy.special.softmax(prediction*100)
    tilted/=tilted.sum()
    return np.r_[0.,tilted]
