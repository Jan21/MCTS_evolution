# NN-labeler track — review memo

## 1. The cleanest claim

One size-free network trained only on boards up to 16x16, used as a
certified-descent labeler, picks the same best candidate as the exact solver
86-93% of the time at *every* board size from 17 to 64 with no cliff — and where
that agreement reaches ~89%, a planner trained on its labels performs the same as
one trained on exact labels.

## 2. What is genuinely unresolved

- **The 82-vs-89% threshold rests on three cells, mostly one seed each**, and
  fidelity co-varies with corpus shrinkage (twins are 63-76% of exact size) and
  robot count, so no cell isolates the causal ingredient.
- **8 robots — the regime the method was built for — is where it fails.** 82.2%
  labels cost 21 points of solve rate; the twin policy net's best epoch was 0 of
  30. Unknown whether better descent (beam, not greedy) would fix it.
- **The 80/96 labels have no downstream test and no planner.** Training above
  32x32 is unprobed (training memory is far worse than inference; a job already
  OOMed at 32), and the 32x32 "NN-everything" deployment run is currently failing
  (4669684/85 NonZeroExitCode; retry 4670119 running).
- **B2 stays negative.** Uncapped descent yielded 0.3-1.3% by-reference
  candidates vs the 13.5% target: the bottleneck was training signal, not search
  budget. So no compute saving is demonstrated anywhere — base vocabulary is
  9-250x cheaper exact at every size measured.
- **Yield thins with scale** (52% kept at 32/64, 48% at 80, 28% at 96), and what
  the dropped instances have in common is uncharacterised.

## 3. Next steps

1. **Size-matched control** — subsample the exact corpora to the twin record
   count and retrain, because fidelity/size/hardness currently move together and
   the dose-response is not yet causal. Payoff: high (secures the headline).
   Cost: small.
2. **Two more seeds on the g32r4 and g24r8 twin cells**, since every caveat in
   FINDINGS 73/74 is a single-seed caveat and the known seed spread is 3.5 pts.
   Payoff: moderate; removes the standing objection. Cost: small.
3. **Try beam or multi-restart descent at 8 robots** instead of greedy
   completion: if fidelity is the causal knob, pushing r8 labels past 89% should
   recover ~20 points downstream. Payoff: high. Cost: medium.
4. **Buy a small uncapped exact B2 seed corpus** at 16x16 and retrain the B2 net
   on it, because the failure was distilled capped data and B2 is the only regime
   where the method can save real compute. Payoff: high if it works. Cost:
   medium-large.
5. **Probe planner-training feasibility above 32x32, then land one downstream
   cell at 40 or 48**, because the 80/96 labels have only a label-side warranty
   and nobody has shown they are trainable-on. Payoff: opens the beyond-oracle
   claim. Cost: small probe, large cell.
