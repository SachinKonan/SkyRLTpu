"""CPU budget probe: TF-IDF/SVD, XGBoost and constrained CVXPY allocation."""
import numpy as np
import cvxpy as cp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBRegressor

NUMERIC=['return_1d','return_5d','return_20d','volatility_20d',
         'relative_volume_20d','treasury_3m_yield','news_count_7d']


def numeric(frame):
    return np.nan_to_num(frame[NUMERIC].to_numpy(dtype=float),nan=0.,posinf=0.,neginf=0.)


def fit(train,valid,rng,*,time_budget_s):
    data=train.loc[np.isfinite(train.target_return.to_numpy(dtype=float))]
    vectorizer=TfidfVectorizer(max_features=12000,min_df=5,dtype=np.float32)
    text=vectorizer.fit_transform(data.headline_text.fillna(''))
    svd=TruncatedSVD(n_components=32,n_iter=3,random_state=1)
    reduced=svd.fit_transform(text)
    x=np.column_stack([numeric(data),reduced])
    learner=XGBRegressor(n_estimators=160,max_depth=3,learning_rate=.03,
        tree_method='hist',device='cpu',n_jobs=4,random_state=1,reg_lambda=20.)
    learner.fit(x,np.clip(data.target_return.to_numpy(dtype=float),-.2,.2))
    assets=sorted(train.asset_id.unique())
    returns=train.pivot(index='session',columns='asset_id',values='return_1d').reindex(columns=assets).fillna(0).to_numpy()
    covariance=np.cov(returns,rowvar=False)+np.eye(len(assets))*1e-5
    return dict(vectorizer=vectorizer,svd=svd,learner=learner,covariance=covariance)


def act(model,obs):
    frame=obs['market']
    text=model['vectorizer'].transform(frame.headline_text.fillna(''))
    x=np.column_stack([numeric(frame),model['svd'].transform(text)])
    pred=np.clip(model['learner'].predict(x),-.02,.02)
    active=frame.price_observed.to_numpy(dtype=float)
    n=len(active);w=cp.Variable(n)
    objective=pred@w-10*cp.quad_form(w,cp.psd_wrap(model['covariance']))-.001*cp.norm1(w-obs['current_weights'][1:])
    problem=cp.Problem(cp.Maximize(objective),[w>=0,w<=.15*active,cp.sum(w)<=1])
    problem.solve(solver='CLARABEL',max_iter=80)
    if w.value is None:return np.r_[1.,np.zeros(n)]
    assets=np.maximum(np.asarray(w.value),0)*active
    assets/=max(1.,assets.sum())
    return np.r_[max(0.,1-assets.sum()),assets]
