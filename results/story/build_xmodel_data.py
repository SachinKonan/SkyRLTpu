import json, gzip, glob, os, math
S = os.path.dirname(os.path.abspath(__file__))
RUNS = {  # tag: (metrics file, member prefix or "", label)
 "full120b": ("full120b_metrics.jsonl","","gpt-oss-120b entropic 64x8"),
 "lrn": ("lrn_metrics.jsonl","qwen/","qwen GRPO 1.5e-4"),
 "ggn": ("ggn_metrics.jsonl","gemma/","gemma GRPO rerun 4e-5"),
 "mgn": ("mgn_metrics.jsonl","muse/","muse GRPO 4e-5"),
 "agn": ("agn_metrics.jsonl","qwen/","qwen GRPO-N (stage A)"),
 "atn": ("atn_metrics.jsonl","qwen/","qwen TTD-N (stage A)"),
 "gtn": ("gtn_metrics.jsonl","gemma/","gemma TTD-N"),
 "mtn": ("mtn_metrics.jsonl","muse/","muse TTD 1.5e-4"),
 "mln": ("mln_metrics.jsonl","muse/","muse GRPO 1.5e-4"),
 "objg": ("obj_grpo_16x32_metrics.jsonl","","gpt-oss-20b GRPO 16x32"),
 "objt": ("obj_ttt_16x32_metrics.jsonl","","gpt-oss-20b entropic 16x32"),
 "ctrl15": ("ctrl15_metrics.jsonl","","gpt-oss-20b entropic 64x8"),
 "a8x64c": ("a8x64c_metrics.jsonl","","gpt-oss-120b entropic 8x64"),
 "tlr": ("tlr_metrics.jsonl","qwen/","qwen entropic 1.5e-4"),
 "gtlr": ("gtlr_metrics.jsonl","gemma/","gemma entropic 1.5e-4"),
 "mttd": ("mttd_metrics.jsonl","muse/","muse TTD 1.5e-4"),
}
out = {}
for tag,(mf,pre,label) in RUNS.items():
    rows=[json.loads(l) for l in open(f"{S}/{mf}")]
    ser=[]; best=None
    for d in rows:
        g=lambda k,dflt=None: d.get(pre+k, d.get(k, dflt))
        bv = d.get("pool/best_value", g("puct/buffer_value/max"))
        c5 = abs(bv) if bv is not None else None
        if c5 is not None: best = c5 if best is None or c5<best else best
        ser.append(dict(step=d.get("step",d.get("global_step")),
            fmt=g("env/all/format"), corr=g("env/all/correctness"),
            rmean=g("env/all/reward/mean",g("env/all/reward")), rmax=g("env/all/reward/max"),
            amax=g("advantage/max"), amin=g("advantage/min"), amean=g("advantage/mean"),
            best=best))
    out[tag]=dict(label=label, series=ser)

# per-rollout categories + delta distributions from trajectory archives
def rparent(pv): return 1.0/(1e-8+abs(pv))
for tag in ("lrn","ggn","mgn","tlr","gtlr","mttd"):
    cats={}
    for f in sorted(glob.glob(f"{S}/{tag}_0*.jsonl.gz")):
        step=int(f.split("_")[-1].split(".")[0])
        n=fail=imp=noop=reg=0; deltas=[]
        for line in gzip.open(f,"rt"):
            r=json.loads(line); n+=1
            if r.get("correctness",0)!=1: fail+=1; continue
            d=float(r["reward"])-rparent(float(r["parent_value"]))
            deltas.append(d)
            if d>0: imp+=1
            elif d<0: reg+=1
            else: noop+=1
        # signed log-magnitude histogram bins: -10..-2 regress, point mass noop, -10..-2 improve
        hb={}
        for d in deltas:
            if d==0: key="0"
            else:
                key=("+" if d>0 else "-")+str(max(-10,min(-1,int(math.floor(math.log10(abs(d)))))))
            hb[key]=hb.get(key,0)+1
        cats[step]=dict(n=n,fail=fail,imp=imp,noop=noop,reg=reg,hist=hb)
    out[tag]["cats"]=cats

# gpt-oss per-rollout categories from local wandb gen&score tables
WB = {
 "full120b": "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss120b_full/tinker_log/*/wandb/latest-run/files/media/table",
 "a8x64c": "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_120b_a8x64_ctrl/tinker_log/*/wandb/latest-run/files/media/table",
 "objg": "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_obj_grpo_16x32/tinker_log/*/wandb/latest-run/files/media/table",
 "objt": "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_obj_ttt_16x32/tinker_log/*/wandb/latest-run/files/media/table",
 "ctrl15": "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss20b_ctrl15/tinker_log/*/wandb/latest-run/files/media/table",
}
for tag, pat in WB.items():
    cats={}
    for f in glob.glob(pat.replace("*", "*", 1) + "/gen&score_train_*.table.json") or glob.glob(glob.glob(pat)[0] + "/gen&score_train_*.table.json") if False else []:
        pass
    dirs = glob.glob(pat)
    if not dirs: continue
    for f in sorted(glob.glob(dirs[0] + "/gen&score_train_*.table.json")):
        step = int(f.split("gen&score_train_")[1].split("_")[0])
        rows = json.load(open(f))["data"]
        n=fail=imp=noop=reg=0; hb={}
        for row in rows:
            n += 1
            if float(row[3]) != 1.0: fail += 1; continue
            d = float(row[2]) - rparent(float(row[6]))
            if d > 0: imp += 1
            elif d < 0: reg += 1
            else: noop += 1
            key = "0" if d == 0 else ("+" if d > 0 else "-") + str(max(-10, min(-1, int(math.floor(math.log10(abs(d)))))))
            hb[key] = hb.get(key, 0) + 1
        cats[step] = dict(n=n, fail=fail, imp=imp, noop=noop, reg=reg, hist=hb)
    out[tag]["cats"] = cats
    print(f"wandb cats {tag}: {len(cats)} steps")

# 120b committed-lineage categories from the final tree snapshot
snaps=sorted(glob.glob("/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss120b_full/tinker_log/*/puct_sampler_step_*.json"))
d=json.load(open(snaps[-1]))
by={}
for s in d["states"]:
    t=s.get("timestep",-1)
    if t<0 or not s.get("parent_values"): continue
    dv = s["value"]-s["parent_values"][-1]   # value=-c5, higher better
    b=by.setdefault(t,dict(n=0,imp=0,noop=0,reg=0))
    b["n"]+=1
    b["imp" if dv>0 else ("reg" if dv<0 else "noop")]+=1
out["full120b"]["lineage"]={str(k):v for k,v in sorted(by.items())}
out["full120b"]["snap_used"]=snaps[-1].split("/")[-1]

json.dump(out,open(f"{S}/xmodel_data.json","w"))
# summary print
for tag in RUNS:
    o=out[tag]; s=o["series"]
    print(f"{tag:9s} steps={len(s):2d} fmt0={s[0]['fmt']:.3f} corr0={s[0]['corr']:.3f} "
          f"fmtE={s[-1]['fmt']:.3f} corrE={s[-1]['corr']:.3f} best={s[-1]['best']:.9f} "
          f"advmaxE={s[-1]['amax'] if s[-1]['amax'] is not None else float('nan'):.2f}")
    if "cats" in o:
        ks=sorted(o["cats"]); a,b=o["cats"][ks[0]],o["cats"][ks[-1]]
        f2=lambda c: f"F{c['fail']/c['n']:.2f}/I{c['imp']/c['n']:.2f}/N{c['noop']/c['n']:.2f}/R{c['reg']/c['n']:.2f}"
        print(f"          cats s{ks[0]}: {f2(a)}  ->  s{ks[-1]}: {f2(b)}")
if "lineage" in out["full120b"]:
    L=out["full120b"]["lineage"]; ks=sorted(L,key=int)
    print("120b lineage steps:", ks[0],"..",ks[-1], "example last:",L[ks[-1]])
