"""Scaling ladder table: every rung, every arm, from the result files.

Arms: sub-goal planner network-free (top_m 0 / 8 / 32, 20 s cap),
matched-time network-free control, and the trained size-free pair (sgnet).
Solve rate is over ALL ladder instances (graded or not); d* is reference only.
"""
import glob, json, os, re, statistics as st
from collections import defaultdict
R = "results/subgoal"
sizes = defaultdict(dict)

def rung(cfg):
    return int(re.match(r"g(\d+)r", cfg).group(1))

for f in glob.glob(f"{R}/ladder/g*_m*.json"):
    d = json.load(open(f)); s = d["summary"]; rows = d["rows"]
    sizes[rung(d["config"])][f"m{d['top_m']}"] = (s["n"], s["solve_rate"], st.mean(r["seconds"] for r in rows))
for f in glob.glob(f"{R}/ladder_matched/g*_tc*.json"):
    d = json.load(open(f)); s = d["summary"]; rows = d["rows"]
    sizes[rung(d["config"])][f"matched"] = (s["n"], s["solve_rate"], st.mean(r["seconds"] for r in rows), d["time_cap"])
TAGS = ("_split", "_sp64", "_sp96c", "_sp96b", "_fast", "_fast2", "_p6", "_p6v6", "_full")   # tagged network arms: split value; pairs from self-play at 64, 96 via 64, 96 direct
net = defaultdict(list); net_split = defaultdict(list); net_sp64 = defaultdict(list); net_sp96c = defaultdict(list); net_sp96b = defaultdict(list); net_fast = defaultdict(list); net_fast2 = defaultdict(list); net_p6 = defaultdict(list); net_p6v6 = defaultdict(list); net_full = defaultdict(list)
arms = {"_split": net_split, "_sp64": net_sp64, "_sp96c": net_sp96c, "_sp96b": net_sp96b, "_fast": net_fast, "_fast2": net_fast2, "_p6": net_p6, "_p6v6": net_p6v6, "_full": net_full}
for f in glob.glob(f"{R}/sgnet/g*.json"):
    if "smoke" in f: continue
    base = os.path.basename(f)[:-5]
    tag = next((t for t in TAGS if base.endswith(t)), "")
    dest = arms[tag] if tag else net
    parts = base[:len(base)-len(tag)].split("_") if tag else base.split("_"); cfg = parts[0]
    off = int(parts[2]) if len(parts) > 2 else 0          # shard files: <cfg>_<lim>_<off>[_split].json
    d = json.load(open(f))
    for name, sy in d["systems"].items():
        for i, r in enumerate(sy["rows"]):
            r["idx"] = off + i                              # ladder instance index (file order)
            dest[rung(cfg)].append(r)
for arm, store in (("net", net), ("net_split", net_split), ("net_sp64", net_sp64), ("net_sp96c", net_sp96c), ("net_sp96b", net_sp96b), ("net_fast", net_fast), ("net_fast2", net_fast2), ("net_p6", net_p6), ("net_p6v6", net_p6v6), ("net_full", net_full)):
    for n, rows in store.items():
        rows = list({r["idx"]: r for r in rows}.values())
        store[n] = rows
        sizes[n][arm] = (len(rows), 100.0 * sum(bool(r["solved"]) for r in rows) / len(rows), st.mean(r["seconds"] for r in rows))

print(f"{'size':>4} | {'m0 (n, %, s)':>22} | {'m8':>22} | {'m32':>22} | {'matched (n,%,s,cap)':>26} | {'net (n,%,s)':>20} | {'net_split (n,%,s)':>20}")
for n in sorted(sizes):
    a = sizes[n]
    def fmt(k, cap=False):
        v = a.get(k)
        if not v: return "-"
        s = f"{v[0]}, {v[1]:.1f}%, {v[2]:.1f}s"
        return s + (f", cap {v[3]:.0f}" if cap and len(v) > 3 else "")
    print(f"{n:>4} | {fmt('m0'):>22} | {fmt('m8'):>22} | {fmt('m32'):>22} | {fmt('matched', True):>26} | {fmt('net'):>20} | {fmt('net_split'):>20}")

# same-subset view: every arm restricted to the env_ids the network arm ran on
print("\nSame instances as the network arm (instance index match), solve %:")
print(f"{'size':>4} | {'n':>3} | {'m0':>6} | {'m8':>6} | {'m32':>6} | {'matched':>8} | {'net':>6} | {'net_split (%, s)':>18} | {'net_sp64 (%, s)':>18} | {'sp96 via 64':>18} | {'sp96 direct':>18} | {'split+fast wrappers':>20} | {'+ axial attention':>20} | {'+ 6-round proposer':>20} | {'+ 6-round value':>20}")
def tagged(store, n, ids):
    rows = [r for r in store.get(n, []) if r["idx"] in ids]
    return f"{100.0*sum(bool(r['solved']) for r in rows)/len(rows):.1f} on {len(rows)}, {st.mean(r['seconds'] for r in rows):.0f}s" if rows else "-"
for n in sorted(sizes):
    ids = {r["idx"] for r in net.get(n, [])}
    if not ids: continue
    def sub(pattern):
        fs = glob.glob(pattern)
        if not fs: return "-"
        rows = [r for r in json.load(open(fs[0]))["rows"] if r["idx"] in ids]
        return f"{100.0*sum(bool(r['solved']) for r in rows)/len(rows):.1f}" if rows else "-"
    cfg = f"g{n}r4"
    print(f"{n:>4} | {len(ids):>3} | {sub(f'{R}/ladder/{cfg}_m0.json'):>6} | {sub(f'{R}/ladder/{cfg}_m8.json'):>6} | "
          f"{sub(f'{R}/ladder/{cfg}_m32.json'):>6} | {sub(f'{R}/ladder_matched/{cfg}_tc*.json'):>8} | {sizes[n]['net'][1]:>6.1f} | "
          + f"{tagged(net_split, n, ids):>18} | {tagged(net_sp64, n, ids):>18} | {tagged(net_sp96c, n, ids):>18} | {tagged(net_sp96b, n, ids):>18} | {tagged(net_fast, n, ids):>20} | {tagged(net_fast2, n, ids):>20} | {tagged(net_p6, n, ids):>20} | {tagged(net_p6v6, n, ids):>20}")


# the real test: the fast configuration on the SAME 100 instances the network-free arms ran
print("\nAll 100 ladder instances per size (network-free arms at a 20 s cap; fast net = both 6-round students, axial, wrappers):")
print(f"{'size':>4} | {'m0 %':>6} | {'m8 %':>6} | {'m32 %':>6} | {'fast net % (n)':>16} | {'m0 s':>6} | {'net s':>6} | {'matched-time net-free % (cap s)':>32}")
for n in sorted(sizes):
    rows = net_full.get(n)
    if not rows: continue
    ids = {r["idx"] for r in rows}
    def arm(k):
        fs = glob.glob(f"{R}/ladder/g{n}r4_{k}.json")
        if not fs: return None
        rr = [r for r in json.load(open(fs[0]))["rows"] if r["idx"] in ids]
        return (100.0 * sum(bool(r["solved"]) for r in rr) / len(rr), st.mean(r["seconds"] for r in rr)) if rr else None
    m0, m8, m32 = arm("m0"), arm("m8"), arm("m32")
    fmt = lambda v: f"{v[0]:6.1f}" if v else "     -"
    net_s = st.mean(r['seconds'] for r in rows)
    # matched-time control: the network-free planner with the iteration cap lifted and the
    # guided arm's mean seconds per instance as the wall-clock cap, same instances
    best = None
    for f in glob.glob(f"{R}/ladder_matched/g{n}r4_tc*.json"):
        d = json.load(open(f)); rr = [r for r in d["rows"] if r["idx"] in ids]
        if len(rr) < len(ids): continue
        tc = float(d["time_cap"])
        if best is None or abs(tc - net_s) < abs(best[0] - net_s):
            best = (tc, 100.0 * sum(bool(r["solved"]) for r in rr) / len(rr))
    mt = f"{best[1]:6.1f} (cap {best[0]:.0f} s)" if best else "-"
    print(f"{n:>4} | {fmt(m0)} | {fmt(m8)} | {fmt(m32)} | {100.0*sum(bool(r['solved']) for r in rows)/len(rows):6.1f} ({len(rows):>3}) | "
          f"{(m0[1] if m0 else float('nan')):6.1f} | {net_s:6.1f} | {mt:>32}")
