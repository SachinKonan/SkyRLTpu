import json, glob, collections, sys
TOL=1e-9
def analyze(name, R):
    rows=[json.loads(l) for l in open(f"{R}/metrics.jsonl")]
    pool=[-r["puct/buffer_value/max"] for r in rows]
    tabs={}
    for d in sorted(glob.glob(f"{R}/wandb/run-*/files/media/table")):
        for f in glob.glob(d+"/gen&score_train_*.table.json"):
            tabs[int(f.split("gen&score_train_")[1].split("_")[0])]=f
    zero=0; line=[]
    for st in sorted(tabs):
        if st>=len(pool): break
        data=json.load(open(tabs[st])); data=data["data"]
        pv=collections.Counter(round(-float(r[6]),9) for r in data)
        inc=pool[st]; k=sum(c for v,c in pv.items() if abs(v-inc)<TOL)
        top3=sorted(pv)[:3]; share=sum(pv[v] for v in top3)/len(data)
        zero+= (k==0); line.append(f"{k:>3}")
        if st in (3,6,9,12,15,18): print(f"   {name} step {st:>2}: incumbent rollouts {k:>3}  distinct parents {len(pv):>2}  best-3 share {share:.2f}  pool best {inc:.9f}")
    print(f"=> {name}: incumbent got ZERO rollouts on {zero}/{len(line)} steps; per-step counts: {' '.join(line)}")
analyze("120b 64x8 elite0 (full)", "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss120b_full/tinker_log/erdos-gptoss120b-full")
analyze("120b 8x64 elite2 (a8x64_ctrl)", "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_120b_a8x64_ctrl/tinker_log/erdos-120b-a8x64-ctrl")
d=glob.glob("/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss120b_distelite/tinker_log/*")
if d: analyze("120b 64x8 elite8+CE (distelite)", d[0])
