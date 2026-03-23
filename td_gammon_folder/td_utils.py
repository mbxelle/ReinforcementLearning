"""
Small helper utilities for reproducibility and scheduling.

These are separated out so the training file stays easier to read.
"""

import random

import numpy as np

from td_config import Config


def set_all_seeds(seed: int):
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)


def epsilon_by_episode(cfg: Config, episode: int) -> float:
    """
    Linearly decay epsilon from epsilon_start to epsilon_end.

    This controls exploration during self-play.

    Early training:
        more randomness helps explore different positions
    Later training:
        lower epsilon helps exploit what the model has learned
    """
    frac = min(1.0, episode / max(1, cfg.episodes - 1))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)