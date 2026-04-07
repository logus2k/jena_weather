"""
Genetic operators: mutation and crossover.

Mutation randomly resets each gene with a given probability.
Crossover creates a child by uniformly sampling each gene from
one of two parents. Both operators apply constraints after
modification to ensure the resulting configuration is valid.
"""

import random
from copy import deepcopy

from src.evolution.search_space import SEARCH_SPACE
from src.evolution.genotype import apply_genotype_constraints


def mutate_genotype(genotype: dict, mutation_rate: float = 0.2) -> dict:
    """Per-gene random reset mutation. Each gene flips with probability mutation_rate."""
    child = deepcopy(genotype)

    for key, values in SEARCH_SPACE.items():
        if random.random() < mutation_rate:
            child[key] = random.choice(values)

    return apply_genotype_constraints(child)


def crossover_genotypes(parent1: dict, parent2: dict) -> dict:
    child = {}
    for key in SEARCH_SPACE.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return apply_genotype_constraints(child)