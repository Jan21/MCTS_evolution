import json, glob, statistics as st, os
def load(size, tag):
    out = {}
    for f in sorted(glob.glob(f"results/subgoal/sgnet/g{size}r4_*.json")):
        base = os.path.basename(f)[:-5]
        if "smoke" in base: continue
        if tag:
            if not base.endswith(tag): continue
            base = base[:-len(tag)]
        elif base.endswith("_split") or base.endswith("_sp64"): continue
        parts = base.split("_"); off = int(parts[2]) if len(parts) > 2 else 0
        for i, r in enumerate(list(json.load(open(f))["systems"].values())[0]["rows"]): out[off + i] = r
    return out
def mv(r):
    m = r.get('moves'); return len(m) if isinstance(m, list) else (m or 0)
for size in (64, 96):
    old, new = load(size, ""), load(size, "_sp64")
    ids = sorted(set(old) & set(new))
    o = {k: old[k] for k in ids}; n = {k: new[k] for k in ids}
    def s(d): return sum(bool(r["solved"]) for r in d.values())
    print(f"== {size}x{size}, {len(ids)} instances: zero-shot pair solved {s(o)}, {st.mean(r['seconds'] for r in o.values()):.0f} s/inst | "
          f"new pair solved {s(n)}, {st.mean(r['seconds'] for r in n.values()):.0f} s/inst")
    gained = [k for k in ids if n[k]["solved"] and not o[k]["solved"]]; lost = [k for k in ids if o[k]["solved"] and not n[k]["solved"]]
    print(f"   gained {gained}  lost {lost}")
    both = [k for k in ids if o[k]["solved"] and n[k]["solved"]]
    print(f"   on the {len(both)} solved by both: seconds old {st.mean(o[k]['seconds'] for k in both):.1f} -> new {st.mean(n[k]['seconds'] for k in both):.1f}; "
          f"plan length old {st.mean(mv(o[k]) for k in both):.1f} -> new {st.mean(mv(n[k]) for k in both):.1f}; "
          f"expansions old {st.mean(o[k]['expansions'] for k in both):.0f} -> new {st.mean(n[k]['expansions'] for k in both):.0f}")
    for k in gained: print(f"   gained idx {k}: old {o[k]['seconds']:.0f}s exp {o[k]['expansions']} | new {n[k]['seconds']:.0f}s exp {n[k]['expansions']} plan length {mv(n[k])}")
