"""figx9: best-C5-so-far for the live fleet (E arms, d-n control, m-meta record run, LR family)."""
import json, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S=sys.argv[1]
RUNS=[("mmeta","m-meta (muse GRPO, meta exp) — PROGRAM BEST","#000000","-"),
      ("lrn","lr-n (qwen GRPO 1.5e-4)","#d94f2b","-"),("tlr","tlr-n (qwen entropic 1.5e-4)","#d94f2b","--"),
      ("lr2","lr2-n (qwen GRPO 4e-4)","#d94f2b",":"),
      ("mgn","m-grpo-n (muse GRPO 4e-5)","#b8860b","-"),("men","m-e-n (muse E)","#b8860b","-."),
      ("mttd","m-ttd-lr-n (muse TTD 1.5e-4)","#b8860b","--"),
      ("ggn","g-grpo-n rerun (gemma 4e-5)","#1a7f6b","-"),("gtlr","g-tlr-n (gemma entropic 1.5e-4)","#1a7f6b","--"),
      ("dn","d-n (qwen piecewise control)","#4a5568","-"),("en","e-n (qwen E)","#d94f2b","-.")]
fig,a=plt.subplots(figsize=(10.4,5.2))
for tag,lab,col,ls in RUNS:
    f=f"{S}/{tag}_metrics.jsonl"
    if not os.path.exists(f): continue
    v=[]; b=None
    for l in open(f):
        d=json.loads(l); x=d.get("pool/best_value")
        if x is None: continue
        b=abs(x) if b is None else min(b,abs(x)); v.append(b)
    if not v: continue
    a.plot(range(len(v)),v,ls,color=col,lw=2.2 if tag=="mmeta" else 1.5,label=f"{lab}  [{len(v)}] {v[-1]:.9f}")
a.axhline(0.380859049,color="#000",lw=0.6,ls=":",alpha=.5)
a.set_ylim(0.38084,0.38100); a.ticklabel_format(useOffset=False,style="plain")
a.set_xlabel("step index"); a.set_ylabel("best C5 so far")
a.set_title("Live fleet — best C5 so far (refreshed 2026-09-01)")
a.legend(fontsize=7.2,frameon=False,ncol=2)
fig.tight_layout(); fig.savefig("figx9.png",bbox_inches="tight"); print("figx9 written")
