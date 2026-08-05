"""Print the path of the banked labeler of record (v2 if it won its gate).

Same rule as gate_v2_and_twins.slurm: argmin agreement at g32r4, tie-break
smaller mean gap; banked v1 if v2 was never gated or lost. Run from
supervised_valuenet/.
"""
import json
import os

V1 = "nn_labeler/banked/prod_v1_s11.ckpt"
V2 = "nn_labeler/banked/prod_v2_s11.ckpt"


def _summary(p):
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def winner():
    v1 = _summary("nn_labeler/results/capgate_g32r4.json")
    v2 = _summary("nn_labeler/results/v2gate_g32r4.json")
    if v2 is None or v1 is None or not os.path.exists(V2):
        return V1
    key = lambda d: (d["argmin_agreement"], -d["gap_mean"])
    return V2 if key(v2) > key(v1) else V1


if __name__ == "__main__":
    print(winner())
