"""Strip ONLY the new instrumentation keys + wall-clock/provenance fields, then
compare before-vs-after byte-level on the canonical JSON serialisation."""
import json, sys, hashlib

NEW_ROW_KEYS = {"accounting", "moves_seq"}     # additive keys from the branch
WALL = {"seconds"}
AGG_WALL = {"mean_seconds", "median_seconds", "total_seconds", "accounting"}
PROTO_DROP = {"date", "command", "results_file"}

def norm(path, drop_backward_moves):
    d = json.load(open(path))
    for k in PROTO_DROP:
        d.get("protocol", {}).pop(k, None)
    for name, sysd in d.get("systems", {}).items():
        agg = sysd.get("aggregate", {})
        for k in list(agg):
            if k in AGG_WALL:
                agg.pop(k)
        for row in sysd.get("rows", []) or []:
            for k in list(row):
                if k in NEW_ROW_KEYS or k in WALL:
                    row.pop(k)
            # backward `moves` is a LIST only under --dump-moves (additive);
            # forward `moves` is the pre-existing integer count -> keep it.
            if drop_backward_moves and isinstance(row.get("moves"), list):
                row.pop("moves")
    return json.dumps(d, sort_keys=True, separators=(",", ":"))

a, b = sys.argv[1], sys.argv[2]
dbm = "--drop-backward-moves" in sys.argv
sa, sb = norm(a, dbm), norm(b, dbm)
ha, hb = hashlib.sha256(sa.encode()).hexdigest(), hashlib.sha256(sb.encode()).hexdigest()
print(f"  {a.split('/')[-1]}  sha256={ha[:16]}")
print(f"  {b.split('/')[-1]}  sha256={hb[:16]}")
print("  VERDICT:", "IDENTICAL" if sa == sb else "*** DIFFERENT ***")
if sa != sb:
    da, db = json.loads(sa), json.loads(sb)
    for nm in da.get("systems", {}):
        ra = da["systems"][nm].get("rows") or []
        rb = db["systems"].get(nm, {}).get("rows") or []
        for i, (x, y) in enumerate(zip(ra, rb)):
            if x != y:
                print(f"    row {i}: {x} != {y}")
                break
        if da["systems"][nm].get("aggregate") != db["systems"].get(nm, {}).get("aggregate"):
            print(f"    aggregate[{nm}]: {da['systems'][nm]['aggregate']}")
            print(f"                 vs {db['systems'][nm]['aggregate']}")
    sys.exit(1)
