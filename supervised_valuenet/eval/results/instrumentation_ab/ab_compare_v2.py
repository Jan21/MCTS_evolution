"""Normalise away every ADDITIVE key of the accounting branch, then compare."""
import json, sys, hashlib
PROTO_DROP = {"date","command","results_file","d_star_placeholder",
              "count_slides","dump_moves"}
ROW_DROP  = {"accounting","moves_seq","seconds"}
AGG_DROP  = {"mean_seconds","median_seconds","total_seconds","accounting",
             "d_star_placeholder"}
def norm(path):
    d = json.load(open(path))
    for k in PROTO_DROP: d.get("protocol",{}).pop(k,None)
    for sysd in d.get("systems",{}).values():
        for k in list(sysd.get("aggregate",{})):
            if k in AGG_DROP: sysd["aggregate"].pop(k)
        for row in sysd.get("rows") or []:
            for k in list(row):
                if k in ROW_DROP: row.pop(k)
            if isinstance(row.get("moves"), list): row.pop("moves")   # backward dump
    return json.dumps(d, sort_keys=True, separators=(",",":"))
a,b = sys.argv[1], sys.argv[2]
sa,sb = norm(a), norm(b)
print(f"  {a.split('/')[-1]:22s} sha={hashlib.sha256(sa.encode()).hexdigest()[:16]}")
print(f"  {b.split('/')[-1]:22s} sha={hashlib.sha256(sb.encode()).hexdigest()[:16]}")
print("  VERDICT:", "IDENTICAL" if sa==sb else "*** DIFFERENT ***")
if sa!=sb:
    da,db=json.loads(sa),json.loads(sb)
    for nm in da.get("systems",{}):
        ra=da["systems"][nm].get("rows") or []; rb=db["systems"].get(nm,{}).get("rows") or []
        for i,(x,y) in enumerate(zip(ra,rb)):
            if x!=y: print(f"    row {i}: {x}\n         != {y}"); break
    sys.exit(1)
