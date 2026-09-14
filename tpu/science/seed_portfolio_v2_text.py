"""CPU reference: training-only TF-IDF and regularized return regression."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


NUMERIC=['return_1d','return_5d','return_20d','volatility_20d',
         'relative_volume_20d','treasury_3m_yield','news_count_7d']


def numeric(frame):
    return np.nan_to_num(frame[NUMERIC].to_numpy(dtype=float),nan=0.,posinf=0.,neginf=0.)


def fit(train,valid,rng,*,time_budget_s):
    good=np.isfinite(train.target_return.to_numpy(dtype=float))
    data=train.loc[good]
    vectorizer=TfidfVectorizer(max_features=12000,min_df=5,ngram_range=(1,2),
                             sublinear_tf=True,dtype=np.float32)
    text=vectorizer.fit_transform(data.headline_text.fillna(''))
    scaler=StandardScaler().fit(numeric(data))
    x=sparse.hstack([text,sparse.csr_matrix(scaler.transform(numeric(data)))],format='csr')
    learner=Ridge(alpha=100.,solver='lsqr',max_iter=150,tol=1e-4)
    learner.fit(x,np.clip(data.target_return.to_numpy(dtype=float),-.2,.2))
    return dict(vectorizer=vectorizer,scaler=scaler,learner=learner)


def act(model,obs):
    frame=obs['market']
    text=model['vectorizer'].transform(frame.headline_text.fillna(''))
    x=sparse.hstack([text,sparse.csr_matrix(model['scaler'].transform(numeric(frame)))],format='csr')
    pred=model['learner'].predict(x)
    available=frame.price_observed.to_numpy(dtype=bool)
    logits=np.clip(pred*50.,-5.,5.)
    scores=np.exp(logits-logits.max())*available
    weights=np.zeros(len(scores)+1)
    if scores.sum()>0:weights[1:]=.9*scores/scores.sum();weights[0]=.1
    else:weights[0]=1.
    return weights
