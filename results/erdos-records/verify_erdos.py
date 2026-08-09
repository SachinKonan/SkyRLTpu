import json, sys, numpy as np
def grader_c5(h_list):
    h=np.array(h_list,dtype=float); n=len(h); dx=2.0/n
    if not np.all(np.isfinite(h)): return None,"non-finite"
    if np.any(h<0) or np.any(h>1): return None,"out of [0,1]"
    t=n/2.0
    if h.sum()!=t:
        h=h*(t/h.sum())
        if np.any(h<0) or np.any(h>1): return None,"out of [0,1] post-norm"
    return float((np.correlate(h,1.0-h,mode="full")*dx).max()), None
for path in sys.argv[1:]:
    rec=json.load(open(path))
    logged=rec["value_full_precision"]; c,err=grader_c5(rec["construction"])
    if err: print("%-28s ERROR %s" % (path.split('/')[-1], err)); continue
    ok = repr(c)==logged
    delta = c-float(logged)
    print("%-30s logged=%s true=%s %s (overclaim %+.3e)" % (
        path.split('/')[-1], logged, repr(c), "VALID" if ok else "INVALID", delta))
