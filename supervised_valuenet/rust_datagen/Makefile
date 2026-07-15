# rust_datagen — build + verification entry points (Agent D).
#
#   make build   release binary (target/release/datagen)
#   make test    full cargo suite (gates 1-3 golden fixtures, io/CLI tests)
#   make smoke   DESIGN §9 cross-engine smoke: pyref dumps -> Rust engine ->
#                differ, 12 cells (16/24/32/64 x r4/r8 x backward/forward)
#   make verify  cargo suite + smoke + the embedded selftest corpus
#   make bench   cross-engine benchmark battery -> BENCH.md numbers
#                (pyref/bench_compare.py; logs under pyref/cache/bench/)
#
# Shared 128-core box: THREADS defaults to 16 — check `uptime` before
# raising it, and use 32 only briefly for end-to-end numbers.

PY ?= /home/p23131/.conda/envs/ph_main/bin/python3
CARGO ?= cargo
THREADS ?= 16

.PHONY: build test smoke verify bench regression

build:
	$(CARGO) build --release -j 16

test:
	$(CARGO) test --release -j 16

smoke: build
	$(PY) pyref/smoke.py --engine rust --workers $(THREADS)

# gate 5: the 12-instance residual-defect regression, three-way
# (python replay + rust replay + defect-candidate rejection assertions)
regression: build
	$(PY) pyref/verify_regression.py

verify: test smoke regression
	./target/release/datagen selftest

bench: build
	$(PY) pyref/bench_compare.py --threads $(THREADS)
