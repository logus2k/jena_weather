"""
Individual representation and constraint enforcement for the GA.

Each individual is a dictionary of hyperparameters sampled from the
search space. Constraints ensure valid architectures: monotonically
decreasing layer widths and optimizer-weight_decay consistency.
"""

import random
from copy import deepcopy

from src.evolution.search_space import SEARCH_SPACE


def apply_constraints(cfg: dict) -> dict:
    """Enforce architectural validity on a configuration dictionary.

    Rules:
      - units2 <= units1, units3 <= units2 (bottleneck structure)
      - weight_decay = 0 when optimizer is 'adam' (only AdamW uses it)
    """
    cfg = deepcopy(cfg)

    if cfg["units2"] > cfg["units1"]:
        cfg["units2"] = cfg["units1"]

    if cfg["units3"] > cfg["units2"]:
        cfg["units3"] = cfg["units2"]

    if cfg["optimizer_name"] == "adam":
        cfg["weight_decay"] = 0.0

    if cfg["n_layers"] == 1:
        cfg["units2"] = min(cfg["units2"], cfg["units1"])
        cfg["units3"] = min(cfg["units3"], cfg["units2"])

    if cfg["n_layers"] == 2:
        cfg["units3"] = min(cfg["units3"], cfg["units2"])

    return cfg


def sample_individual() -> dict:
    cfg = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    return apply_constraints(cfg)