"""QC gate 2: by-reference share of a B2 label file.

A candidate is 'by reference' when its helper stands somewhere that is NOT any
robot's START position -- i.e. the plan is reusing a robot it has already
placed, which is exactly what lever B2 added. Robot starts are target_robot[0]
plus every helpers[i][0]; there is no `robot_positions` field in the schema.
Expected roughly 5-20%; base measured 13% before this campaign.
"""
import json, sys
n = byref = 0
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    r = json.loads(line)
    n += 1
    starts = {tuple(r["target_robot"][0])}
    starts |= {tuple(h[0]) for h in (r.get("helpers") or [])}
    if tuple(r["cand_helper"][0]) not in starts:
        byref += 1
print(f"QC by-reference share: {byref}/{n} = {byref/n*100:.1f}%" if n else "QC: empty")
