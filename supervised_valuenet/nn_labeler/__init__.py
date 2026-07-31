"""NN-labeler track: size-parametric ports of the backward value net's data path.

Everything here takes the grid side `n` as an ARGUMENT. Nothing in this package
reads RR_GRID / RR_ROBOTS / RR_ENV_DIR at import time, so one process (and one
net) can handle several board configurations at once. See PLANS.md S0.1/S0.2.
"""
