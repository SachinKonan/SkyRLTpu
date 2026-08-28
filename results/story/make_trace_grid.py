import gzip, json, math, statistics as st, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S="/tmp/claude-374192/-n-fs-vision-mix-sk7524-SkyRLTpu/7702cd20-fe25-42c0-871b-c38f00057807/scratchpad"
OUT="/n/fs/vision-mix/sk7524/SkyRLTpu-league/results/story"
CAP=1e6

def solve_beta(r, delta=math.log(2)):
    if len(r)<2: return 0.0
    t=torch.tensor(list(r),dtype=torch.float64); K=math.log(len(r)); lo,hi=0.0,1.0
    def kl(b):
        lg=b*(t-t.max()); lq=lg-torch.logsumexp(lg,0)
        return float((torch.exp(lq)*(lq+K)).sum())
    while hi<CAP and kl(hi)<delta: hi*=2
    if kl(hi)<delta: return float(hi)
    for _ in range(70):
        mid=(lo+hi)/2
        (lo,hi)=(mid,hi) if kl(mid)<delta else (lo,mid)
    return float(hi)

def stats(fn):
    rows=[json.loads(l) for l in gzip.open(f"{S}/{fn}","rt")]
    G={}
    for r in rows: G.setdefault(r["seed_idx"],[]).append(r)
    nf=nn=ni=nr=0; imps=[]; betas=[]; capped=0
    for rs in G.values():
        r=np.array([x["reward"] for x in rs])
        seed=[x.get("initial_raw_score") for x in rs if x.get("initial_raw_score") is not None]
        if not seed: continue
        rpar=1.0/(1e-8+abs(seed[0])); valid=r>0.01
        if valid.sum()>=2:
            b=solve_beta(r[valid]); betas.append(b); capped += (b>=CAP*0.999)
        for x,v in zip(r,valid):
            if not v: nf+=1; continue
            d=x-rpar
            if abs(d)<1e-12: nn+=1
            elif d>0: ni+=1; imps.append(d)
            else: nr+=1
    n=max(1,nf+nn+ni+nr)
    return dict(fail=100*nf/n, noop=100*nn/n, imp=100*ni/n, reg=100*nr/n,
                med_imp=(st.median(imps) if imps else np.nan),
                med_beta=(float(np.median(betas)) if betas else np.nan),
                cap=(100*capped/len(betas) if betas else np.nan))

SERIES={
 "qwen": [("qwen_e0.jsonl.gz",0,"a"),("qwen2_s1.jsonl.gz",1,"a"),("qwen2_s2.jsonl.gz",2,"a"),
          ("qwen_e3.jsonl.gz",3,"a"),
          ("qwen_s11.jsonl.gz",11,"b"),("qwen_t12.jsonl.gz",12,"b"),
          ("qwen_s13.jsonl.gz",13,"b"),("qwen_t14.jsonl.gz",14,"b")],
 "gemma":[("gemma_t05.jsonl.gz",5,"a"),("gemma_s6.jsonl.gz",6,"a"),("gemma_s7.jsonl.gz",7,"a"),
          ("gemma_s8.jsonl.gz",8,"a"),("gemma_t09.jsonl.gz",9,"a"),("gemma_s10.jsonl.gz",10,"a"),
          ("gemma_s11.jsonl.gz",11,"a"),("gemma_t12.jsonl.gz",12,"a"),
          ("gemma_s13.jsonl.gz",13,"a"),("gemma_t14.jsonl.gz",14,"a")],
}
data={}
for m,items in SERIES.items():
    pts=[]
    for fn,stp,seg in items:
        if not os.path.exists(f"{S}/{fn}"): continue
        d=stats(fn); d["step"]=stp; d["seg"]=seg; pts.append(d)
    data[m]=sorted(pts,key=lambda d:d["step"])
    print(m, [(p["step"],round(p["fail"],1),round(p["noop"],1)) for p in data[m]])

QW,GM = "#14867c","#c2410c"
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
PANELS=[("fail","failures  %",None),("noop","no-ops: same score  %",None),
        ("impreg","improve / regress  %",None),
        ("med_imp","median improvement",True),("med_beta","median β",True),
        ("cap","groups at β ceiling  %",None)]
fig,axes=plt.subplots(2,len(PANELS),figsize=(21,7.0))
fig.patch.set_facecolor("white")
for row,(m,c) in enumerate([("qwen",QW),("gemma",GM)]):
    pts=data[m]
    # plot against POSITION, not step number: draw one continuous line and label
    # the ticks with the real steps, so the archive gap doesn't break the curve.
    x=list(range(len(pts)))
    labs=[str(p["step"]) for p in pts]
    segs=[p["seg"] for p in pts]
    bound=next((i for i in range(1,len(segs)) if segs[i]!=segs[i-1]), None)
    for col,(key,ylab,logy) in enumerate(PANELS):
        ax=axes[row][col]
        if key=="impreg":
            ax.plot(x,[p["imp"] for p in pts],"-",color=c,lw=2,label="improve")
            ax.plot(x,[p["reg"] for p in pts],"-",color=c,lw=1.6,alpha=.45,label="regress")
            for xi,p,sg in zip(x,pts,segs):
                mk_="o" if sg=="a" else "s"
                ax.plot([xi],[p["imp"]],mk_,color=c,ms=4.5)
                ax.plot([xi],[p["reg"]],mk_,color=c,ms=4.5,alpha=.45)
        else:
            ax.plot(x,[p[key] for p in pts],"-",color=c,lw=2)
            for xi,p,sg in zip(x,pts,segs):
                ax.plot([xi],[p[key]],"o" if sg=="a" else "s",color=c,ms=4.5)
        if bound is not None:
            ax.axvline(bound-0.5,color="#94a3b8",ls=":",lw=1.1)
        if logy: ax.set_yscale("log")
        if key=="cap": ax.set_ylim(-4,104)
        if key in ("fail","noop"): ax.set_ylim(-3,80)
        if key=="impreg":
            ax.set_ylim(-3,80)
            if row==0: ax.legend(frameon=False,fontsize=8,loc="upper left")
        if row==0: ax.set_title(ylab,fontsize=11,fontweight="bold")
        if col==0: ax.set_ylabel(m,fontsize=13,fontweight="bold",color=c,labelpad=10)
        ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8.5)
        ax.set_xlabel("training step",fontsize=9.5)
axes[0][4].axhline(CAP,color="#9f1239",ls=":",lw=1.3)
axes[1][4].axhline(CAP,color="#9f1239",ls=":",lw=1.3)
axes[0][4].text(0.1,CAP,"solver ceiling",fontsize=8,color="#9f1239",va="bottom")
fig.suptitle("How a group's reward distribution changes over training  —  qwen (top), gemma (bottom)",
             fontsize=13.5,y=.985)
fig.text(.5,.012,"Plotted against position rather than step number, so the curves read continuously. Qwen is two runs joined at the dotted line: circles = steps 0-3 at LR 4e-4, squares = steps 11-14 at LR 1.5e-4. Gemma is one continuous run at 4e-5.",
         ha="center",fontsize=9,color="#64748b")
fig.tight_layout(rect=[0,.035,1,.955])
fig.savefig(f"{OUT}/fig_trace_grid.png",dpi=140,facecolor="white")
print("saved", os.path.getsize(f"{OUT}/fig_trace_grid.png")//1024,"KB")
