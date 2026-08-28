import gzip, json, math, statistics as st
import numpy as np, torch
S="/tmp/claude-374192/-n-fs-vision-mix-sk7524-SkyRLTpu/7702cd20-fe25-42c0-871b-c38f00057807/scratchpad"

def solve_beta(r, delta=math.log(2), cap=1e6):
    t=torch.tensor(list(r),dtype=torch.float64); K=math.log(len(r)); lo,hi=0.0,1.0
    def kl(b):
        lg=b*(t-t.max()); lq=lg-torch.logsumexp(lg,0)
        return float((torch.exp(lq)*(lq+K)).sum())
    if len(r)<2: return 0.0
    while hi<cap and kl(hi)<delta: hi*=2
    if kl(hi)<delta: return float(hi)          # cap-bound (one bit unattainable)
    for _ in range(80):
        mid=(lo+hi)/2
        (lo,hi)=(mid,hi) if kl(mid)<delta else (lo,mid)
    return float(hi)

def grpo(r):
    r=np.array(r); return r-r.mean()

def ttd(r):
    r=np.array(r); b=solve_beta(r)
    e=np.exp(np.clip(b*(r-r.max()),-700,0)); Z=(e.sum()-e)/(len(r)-1)
    return e/(Z+1e-300)-1.0, b

def v2(r, rpar, valid):
    r=np.array(r); v=r[valid]
    b=solve_beta(v) if valid.sum()>=2 else 0.0
    return np.clip(b*(r-rpar), -1.0, 10.0), b

def v3(r, rpar, valid, scale, eps):
    r=np.array(r); imp=r-rpar
    A=np.clip(imp/scale, -1.0, 3.0)
    A[np.abs(imp)<eps]=0.0
    A[~valid]=-1.0
    return A

def load(nm):
    rows=[json.loads(l) for l in gzip.open(f"{S}/{nm}.jsonl.gz","rt")]
    G={}
    for r in rows: G.setdefault(r["seed_idx"],[]).append(r)
    return G

def groupinfo(rs):
    r=np.array([x["reward"] for x in rs])
    seed=[x.get("initial_raw_score") for x in rs if x.get("initial_raw_score") is not None]
    rpar=1.0/(1e-8+abs(seed[0])) if seed else None
    valid=r>0.01
    return r, rpar, valid

out={}
# ---------- per-step summary + EMA scale trace ----------
trace=[]
for nm,lab,step in [("qwen_e0","qwen",0),("qwen_e3","qwen",3),("qwen_t12","qwen",12),("qwen_t14","qwen",14),
                    ("gemma_t05","gemma",5),("gemma_t09","gemma",9),("gemma_t12","gemma",12),("gemma_t14","gemma",14)]:
    G=load(nm); imps=[]; nf=nn=ni=nr=0; betas=[]; capped=0
    for k,rs in G.items():
        r,rpar,valid=groupinfo(rs)
        if rpar is None: continue
        b=solve_beta(r[valid]) if valid.sum()>=2 else 0.0
        betas.append(b); capped += (b>=1e6*0.999)
        for x,v in zip(r,valid):
            if not v: nf+=1; continue
            d=x-rpar
            if abs(d)<1e-12: nn+=1
            elif d>0: ni+=1; imps.append(d)
            else: nr+=1
    n=nf+nn+ni+nr
    trace.append(dict(model=lab, step=step, fail=100*nf/n, noop=100*nn/n, imp=100*ni/n, reg=100*nr/n,
                      med_imp=(st.median(imps) if imps else 0.0),
                      med_beta=float(np.median(betas)), cap_frac=100*capped/max(1,len(betas))))
out["trace"]=trace

# ---------- worked example groups ----------
def worked(nm, pick, label, scale, eps=1e-12):
    G=load(nm)
    r,rpar,valid=groupinfo(G[pick])
    Ag=grpo(r); At,bt=ttd(r); A2,b2=v2(r,rpar,valid); A3=v3(r,rpar,valid,scale,eps)
    idx=np.argsort(-r)
    rows=[]
    for i in idx:
        d=r[i]-rpar
        kind=("fail" if not valid[i] else "noop" if abs(d)<1e-12 else "improve" if d>0 else "regress")
        rows.append(dict(reward=float(r[i]), delta=float(d), kind=kind,
                         grpo=float(Ag[i]), ttd=float(At[i]), v2=float(A2[i]), v3=float(A3[i])))
    return dict(label=label, n=len(r), nfail=int((~valid).sum()), rpar=float(rpar),
                beta_ttd=bt, beta_v2=b2, scale=scale,
                mean_r=float(r.mean()), rows=rows,
                counts={k:sum(1 for x in rows if x["kind"]==k) for k in ("fail","noop","improve","regress")})

# choose representative groups
def pick_group(nm, want):
    G=load(nm); best=None
    for k,rs in G.items():
        r,rpar,valid=groupinfo(rs)
        if rpar is None: continue
        nf=int((~valid).sum())
        score=abs(nf-want)
        if best is None or score<best[0]: best=(score,k)
    return best[1]

ex=[]
ex.append(worked("qwen_t12", pick_group("qwen_t12",3),  "Qwen · step 12 · converged", scale=2e-8))
ex.append(worked("gemma_t12",pick_group("gemma_t12",8), "Gemma · step 12", scale=2e-7))
ex.append(worked("gemma_t12",pick_group("gemma_t12",16),"Gemma · step 12 · half the group failed", scale=2e-7))
ex.append(worked("qwen_e3",  pick_group("qwen_e3",4),   "Qwen · step 3 · still exploring", scale=4e-6))
out["examples"]=ex

# ---------- beta scenarios (synthetic, to isolate each regime) ----------
base=1.0/(1e-8+0.3808616)
def scen(name, rewards, note):
    r=np.array(rewards); v=r>0.01
    b_all=solve_beta(r); b_val=solve_beta(r[v]) if v.sum()>=2 else 0.0
    return dict(name=name, note=note, n=len(r), nfail=int((~v).sum()),
                beta_all=b_all, beta_valid=b_val,
                capped_all=b_all>=1e6*0.999, capped_valid=b_val>=1e6*0.999)
def mk(nv, nf, spread, seed=0):
    rng=np.random.default_rng(seed)
    return list(base+rng.uniform(-spread,spread,nv))+[0.0]*nf
scens=[
 scen("few failures, tiny spread", mk(29,3,1e-8,1), "the normal converged case"),
 scen("half the group failed",     mk(16,16,1e-8,2), "failure separation alone = one bit"),
 scen("most failed",               mk(10,22,1e-8,3), "budget spent before valid samples matter"),
 scen("no failures, tiny spread",  mk(32,0,1e-8,4), "all budget available for discovery"),
 scen("20 identical no-ops + 1 improver", [base]*20+[base+7e-8]+[0.0]*11, "top tie is small -> solvable"),
 scen("all valid tied exactly",    [base]*24+[0.0]*8, "one bit UNATTAINABLE -> solver cap"),
 scen("18 of 24 valid tied at top",[base+1e-7]*18+[base]*6+[0.0]*8, "tie covers >half -> cap"),
]
out["beta_scenarios"]=scens

json.dump(out, open(f"{S}/blog_data.json","w"), indent=1)
print("wrote blog_data.json")
for t in trace: print(f"  {t['model']:<6} step {t['step']:>2}: fail {t['fail']:5.1f}% noop {t['noop']:5.1f}% "
                      f"imp {t['imp']:5.1f}% reg {t['reg']:5.1f}%  med_imp {t['med_imp']:.2e}  "
                      f"med_beta {t['med_beta']:.3g} cap {t['cap_frac']:.0f}%")
print()
for s in scens: print(f"  {s['name']:<34} fails {s['nfail']:>2}  beta(all) {s['beta_all']:>11.4g}"
                      f"  beta(valid-only) {s['beta_valid']:>11.4g}  {'CAP' if s['capped_valid'] else ''}")
