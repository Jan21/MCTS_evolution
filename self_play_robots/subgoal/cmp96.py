import json, glob, statistics as st, os
def rows(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        parts = os.path.basename(f)[:-5].replace("_split", "").split("_")
        off = int(parts[2])
        for i, r in enumerate(list(json.load(open(f))["systems"].values())[0]["rows"]): out[off + i] = r
    return out
full = {k: v for k, v in rows("results/subgoal/sgnet/g96r4_2_*.json").items()}
split = rows("results/subgoal/sgnet/g96r4_2_*_split.json")
full = {k: v for k, v in rows("results/subgoal/sgnet/g96r4_2_[0-9]*.json").items() if k in split}
# rows() with the wildcard picks up _split files too; rebuild full from non-split files only
full = {}
for f in sorted(glob.glob("results/subgoal/sgnet/g96r4_2_*.json")):
    if f.endswith("_split.json"): continue
    off = int(os.path.basename(f)[:-5].split("_")[2])
    for i, r in enumerate(list(json.load(open(f))["systems"].values())[0]["rows"]): full[off + i] = r
def summ(name, d):
    print(f"{name}: solved {sum(bool(r['solved']) for r in d.values())}/{len(d)}, {st.mean(r['seconds'] for r in d.values()):.0f} s/inst, "
          f"sum {sum(r['seconds'] for r in d.values()):.0f} s, expansions {st.mean(r['expansions'] for r in d.values()):.0f}")
summ("full pair ", full); summ("split val ", split)
hard = [k for k in full if full[k]["seconds"] > 100]
print(f"hard ({len(hard)} instances, full pair > 100 s): full {st.mean(full[k]['seconds'] for k in hard):.0f} s -> split {st.mean(split[k]['seconds'] for k in hard):.0f} s; "
      f"solved full {sum(bool(full[k]['solved']) for k in hard)}, split {sum(bool(split[k]['solved']) for k in hard)}; "
      f"expansions full {st.mean(full[k]['expansions'] for k in hard):.0f}, split {st.mean(split[k]['expansions'] for k in hard):.0f}")
easy = [k for k in full if k not in hard]
print(f"easy ({len(easy)}): full {st.mean(full[k]['seconds'] for k in easy):.1f} s -> split {st.mean(split[k]['seconds'] for k in easy):.1f} s")
print("per-instance agreement on solved:", sum(bool(full[k]["solved"]) == bool(split[k]["solved"]) for k in full), "/", len(full))
for k in hard:
    a = full[k]["accounting"]; b = split[k]["accounting"]
    print(f"  idx {k}: full {full[k]['seconds']:.0f}s exp {full[k]['expansions']} val_calls {a.get('nn_value_calls')} solved {full[k]['solved']} | split {split[k]['seconds']:.0f}s exp {split[k]['expansions']} val_calls {b.get('nn_value_calls')} solved {split[k]['solved']}")
