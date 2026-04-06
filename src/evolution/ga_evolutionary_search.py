"""
Genetic algorithm evolutionary search loop.

Implements a generational GA with tournament selection, uniform
crossover, per-gene mutation, and elitism. Each generation evaluates
all individuals by training a GRU model end-to-end and measuring
validation MAE. The best configuration across all generations is
returned along with the full evolution history.
"""

import random

from src.evolution.genotype import sample_genotype
from src.evolution.operators import mutate_genotype, crossover_genotypes
from src.evolution.fitness import evaluate_individual


def tournament_selection(population_results, k=3):
    """Select the best individual from k random candidates (lower fitness wins)."""
    candidates = random.sample(population_results, k=min(k, len(population_results)))
    candidates = sorted(candidates, key=lambda x: x["fitness"])
    return candidates[0]["cfg"]


def run_evolutionary_search(
    df_train,
    df_val,
    final_feature_cols,
    target_idx,
    population_size=8,
    generations=5,
    mutation_rate=0.2,
    elitism=2,
    lookback=120,
    horizon=24,
    epochs=10,
    verbose=0,
):
    population = [sample_genotype() for _ in range(population_size)]
    history = []

    best_result = None

    for gen in range(generations):
        population_results = []

        for cfg in population:
            result = evaluate_individual(
                cfg=cfg,
                df_train=df_train,
                df_val=df_val,
                final_feature_cols=final_feature_cols,
                target_idx=target_idx,
                lookback=lookback,
                horizon=horizon,
                epochs=epochs,
                verbose=verbose,
            )
            population_results.append(result)

        population_results = sorted(population_results, key=lambda x: x["fitness"])
        history.append(population_results)

        if best_result is None or population_results[0]["fitness"] < best_result["fitness"]:
            best_result = population_results[0]

        print(
            f"Generation {gen + 1}/{generations} - "
            f"best fitness: {population_results[0]['fitness']:.6f}"
        )

        next_population = [res["cfg"] for res in population_results[:elitism]]

        while len(next_population) < population_size:
            parent1 = tournament_selection(population_results)
            parent2 = tournament_selection(population_results)
            child = crossover_genotypes(parent1, parent2)
            child = mutate_genotype(child, mutation_rate=mutation_rate)
            next_population.append(child)

        population = next_population

    return best_result, history