"""figx8: why the 120b 64x8 entropic run sat flat for 9 steps -- the incumbent was never re-expanded.
Reads the run's per-step wandb gen tables; writes figx8.png + plateau_120b.json."""
import json, glob, collections, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss120b_full/tinker_log/erdos-gptoss120b-full"
rows=[json.loads(l) for l in open(f"{R}/metrics.jsonl")]
pool_best=[-r["puct/buffer_value/max"] for r in rows]           # pool at START of step i
tabs={}
for d in sorted(glob.glob(f"{R}/wandb/run-*/files/media/table")):
    for f in glob.glob(d+"/gen&score_train_*.table.json"):
        tabs[int(f.split("gen&score_train_")[1].split("_")[0])]=f
TOL=1e-9; steps=sorted(tabs); rec=[]
for st in steps:
    data=json.load(open(tabs[st]))["data"]
    pv=collections.Counter(round(-float(r[6]),9) for r in data)
    kids=[1.0/float(r[2])-1e-8 for r in data if float(r[3])==1.0]
    inc=pool_best[st]; inc_kids=sum(c for v,c in pv.items() if abs(v-inc)<TOL)
    top3=sorted(pv)[:3]
    rec.append(dict(step=st, pool_best=inc, best_parent_sampled=min(pv), incumbent_children=inc_kids,
                    distinct_parents=len(pv), share_top3=sum(pv[v] for v in top3)/len(data),
                    best_child=min(kids) if kids else None))
json.dump(rec, open("plateau_120b.json","w"), indent=1)
fig,(a1,a2)=plt.subplots(2,1,figsize=(8.6,5.6),sharex=True,gridspec_kw=dict(height_ratios=[3,1.6]))
x=[r["step"] for r in rec]
a1.plot(x,[r["pool_best"] for r in rec],color="#8659c9",lw=2,label="pool best (incumbent)")
a1.plot(x,[r["best_parent_sampled"] for r in rec],"o--",color="#b8860b",ms=4,lw=1.2,label="best PARENT actually sampled this step")
a1.plot(x,[r["best_child"] for r in rec],"s:",color="#1a7f6b",ms=4,lw=1.2,label="best CHILD produced this step")
for r in rec:
    if r["incumbent_children"]==0: a1.axvspan(r["step"]-0.5,r["step"]+0.5,color="#c0392b",alpha=0.07)
a1.set_ylim(0.38085,0.38105); a1.ticklabel_format(useOffset=False,style="plain")
a1.set_ylabel("C5"); a1.legend(fontsize=8,frameon=False,loc="upper right")
a1.set_title("gpt-oss-120b entropic 64x8 (elite=0): shaded = steps where the incumbent got ZERO rollouts")
a2.bar(x,[r["incumbent_children"] for r in rec],color="#8659c9",label="rollouts on the incumbent")
a2.plot(x,[100*r["share_top3"] for r in rec],"k.-",lw=1,label="% of rollouts on best-3 parents")
a2.set_ylabel("count / %"); a2.set_xlabel("step"); a2.legend(fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig("figx8.png",bbox_inches="tight")
print("\n".join(f"step {r['step']:>2}: pool {r['pool_best']:.9f} best-parent-sampled {r['best_parent_sampled']:.9f} incumbent-kids {r['incumbent_children']:>2} distinct {r['distinct_parents']:>2} top3-share {r['share_top3']:.2f} best-child {r['best_child']:.9f}" for r in rec))
