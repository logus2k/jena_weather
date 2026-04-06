import random
from copy import deepcopy

from src.evolution.search_space import SEARCH_SPACE


def apply_genotype_constraints(genotype: dict) -> dict:
    g = deepcopy(genotype)

    if g["units2"] > g["units1"]:
        g["units2"] = g["units1"]

    if g["units3"] > g["units2"]:
        g["units3"] = g["units2"]

    if g["optimizer_name"] == "adam":
        g["weight_decay"] = 0.0

    if g["n_layers"] == 1:
        g["units2"] = min(g["units2"], g["units1"])
        g["units3"] = min(g["units3"], g["units2"])

    if g["n_layers"] == 2:
        g["units3"] = min(g["units3"], g["units2"])

    return g


def sample_genotype() -> dict:
    genotype = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    return apply_genotype_constraints(genotype)