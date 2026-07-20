"""Analyze anytime vs plain backward comparison JSONs (read-only on repo).

Slices the plain-450 baseline rows to the anytime run's instance count so all
comparisons are order-matched on identical instances.
"""
import json
from collections import Counter, defaultdict

BASE = "/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/eval/results"
plain = json.load(open(f"{BASE}/comparison_backward.json"))
anyt = json.load(open(f"{BASE}/comparison_backward_anytime.json"))

pr_full = plain["systems"]["backward subgoal planner"]["rows"]
ar = anyt["systems"]["backward subgoal planner (anytime realization-checked)"]["rows"]
N = len(ar)
pr = pr_full[:N]
print(f"n anytime rows = {N}; plain sliced to first {N} of {len(pr_full)}")
print("sha plain:", plain["protocol"]["instances_sha256"][:16],
      "anytime:", anyt["protocol"]["instances_sha256"][:16])
assert all(p["env_id"] == a["env_id"] and p["d_star"] == a["d_star"]
           for p, a in zip(pr, ar)), "row order mismatch"
print("rows aligned by order: OK\n")

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def headline(rows, name):
    n = len(rows)
    sol = [r for r in rows if r["solved"]]
    mk = "realized_strict" if "realized_strict" in rows[0] else "moves"
    print(f"[{name}] solved {len(sol)}/{n} ({100*len(sol)/n:.1f}%) "
          f"regret={mean([r['regret'] for r in sol]):.3f} "
          f"opt%={100*sum(1 for r in sol if r['regret']==0)/len(sol):.1f} "
          f"strict_moves={mean([r[mk] for r in sol]):.2f} "
          f"exp={mean([r['expansions'] for r in rows]):.2f} "
          f"s/inst={mean([r['seconds'] for r in rows]):.3f} "
          f"plan_found={sum(1 for r in rows if r.get('plan_found'))}")

headline(pr_full, "plain-450 (reference)")
headline(pr, f"plain-{N} (matched)  ")
headline(ar, f"anytime-{N}          ")

rej = [r.get("plans_rejected", 0) for r in ar]
print("\nplans_rejected: >0 on", sum(1 for x in rej if x > 0), f"of {N} instances; max =",
      max(rej), "; mean =", f"{mean(rej):.2f}")
hist = Counter(min(x, 10) if x <= 10 else (11 if x <= 50 else 12) for x in rej)
lab = {**{i: str(i) for i in range(11)}, 11: "11-50", 12: ">50"}
print("rejected histogram:", {lab[k]: v for k, v in sorted(hist.items())})
print("mean rejected | solved:", f"{mean([r['plans_rejected'] for r in ar if r['solved']]):.2f}",
      "| failed:", f"{mean([r['plans_rejected'] for r in ar if not r['solved']]) or 0:.2f}")

def b(ds):
    return "1-3" if ds <= 3 else "4-6" if ds <= 6 else "7-9" if ds <= 9 else "10+"

print("\nper-bin (matched instances, plain -> anytime):")
for key in ["1-3", "4-6", "7-9", "10+"]:
    for name, rows in [("plain", pr), ("anyt ", ar)]:
        rs = [r for r in rows if b(r["d_star"]) == key]
        if not rs:
            print(f"  {key:>3} {name}: n=0")
            continue
        sol = [r for r in rs if r["solved"]]
        mr = mean([r["regret"] for r in sol])
        print(f"  {key:>3} {name}: n={len(rs)} solved={len(sol)} "
              f"({100*len(sol)/len(rs):.0f}%) regret={mr if mr is None else round(mr,3)} "
              f"exp={mean([r['expansions'] for r in rs]):.1f} "
              f"sec={mean([r['seconds'] for r in rs]):.3f} "
              f"rej={mean([r.get('plans_rejected',0) for r in rs]):.1f}")

# transitions on matched instances
pf = [i for i in range(N) if not pr[i]["solved"]]
pf_as = [i for i in pf if ar[i]["solved"]]
ps_af = [i for i in range(N) if pr[i]["solved"] and not ar[i]["solved"]]
both = [i for i in range(N) if pr[i]["solved"] and ar[i]["solved"]]
print(f"\nplain-failed among first {N}: {len(pf)}; rescued by anytime: {len(pf_as)} "
      f"({100*len(pf_as)/len(pf):.0f}% of failures)")
print(f"plain-solved but anytime-FAILED (determinism check): {len(ps_af)}")
diff_moves = [i for i in both if pr[i]["realized_strict"] != ar[i]["realized_strict"]]
print(f"both-solved: {len(both)}; strict-move mismatches: {len(diff_moves)}")
for i in ps_af[:5]:
    print("  PS/AF:", pr[i]["env_id"], "d*", pr[i]["d_star"], "rej", ar[i]["plans_rejected"])
for i in diff_moves[:5]:
    print("  MOVDIFF:", pr[i]["env_id"], pr[i]["realized_strict"], "->", ar[i]["realized_strict"])

if pf_as:
    print("\nrescued instances: mean regret", f"{mean([ar[i]['regret'] for i in pf_as]):.3f}",
          "| mean rejected", f"{mean([ar[i]['plans_rejected'] for i in pf_as]):.1f}",
          "| mean exp", f"{mean([ar[i]['expansions'] for i in pf_as]):.1f}",
          "| mean sec", f"{mean([ar[i]['seconds'] for i in pf_as]):.2f}")
    print("rescued d* profile:", dict(sorted(Counter(b(ar[i]["d_star"]) for i in pf_as).items())))
    print("rescued optimal (regret==0):", sum(1 for i in pf_as if ar[i]["regret"] == 0))

fails = [r for r in ar if not r["solved"]]
print(f"\nresidual failures: {len(fails)} of {N} "
      f"({100*len(fails)/N:.1f}%)")
if fails:
    print("  d* bin profile:", dict(sorted(Counter(b(r["d_star"]) for r in fails).items())))
    print("  d* values:", dict(sorted(Counter(r["d_star"] for r in fails).items())))
    print("  plan_found (fallback first_failed):", sum(1 for r in fails if r["plan_found"]))
    exp_hist = Counter(r["expansions"] for r in fails)
    print("  expansions:", dict(sorted(exp_hist.items())))
    print("  exhausted budget (exp>=1200):", sum(1 for r in fails if r["expansions"] >= 1200))
    print("  mean rejected:", f"{mean([r['plans_rejected'] for r in fails]) or 0:.1f}",
          "| max rejected:", max(r["plans_rejected"] for r in fails))
    print("  mean seconds:", f"{mean([r['seconds'] for r in fails]):.2f}")

sol = [r for r in ar if r["solved"]]
print("\nanytime cost profile: mean exp | solved:", f"{mean([r['expansions'] for r in sol]):.2f}",
      "| failed:", f"{mean([r['expansions'] for r in fails]) if fails else 0:.1f}")
print("max seconds single instance:", f"{max(r['seconds'] for r in ar):.1f}",
      "| sum wall:", f"{sum(r['seconds'] for r in ar)/60:.1f} min")
print("\nPhase A bar check (on these instances): solve>=97.5%? "
      f"{100*len(sol)/N:.1f}%  regret<=0.35? {mean([r['regret'] for r in sol]):.3f}  "
      f"exp {mean([r['expansions'] for r in ar]):.1f} vs forward 36")
