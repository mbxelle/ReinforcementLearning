"""
TD(lambda) update logic.

This file contains the actual weight-update rule used during training.
Splitting it out makes the training loop easier to read.
"""

from typing import Dict

import numpy as np

from td_model import ValueNet


def td_lambda_update(
    model: ValueNet,
    traces: Dict[str, np.ndarray],
    x_t: np.ndarray,
    target: float,
    alpha: float,
    gamma: float,
    lam: float,
) -> float:
    """
    Perform one semi-gradient TD(lambda) update.

    Core chapter-6 idea:
        delta_t = target_t - V(S_t)

    Where:
    - V(S_t) is the current network prediction for the current state
    - target_t is the bootstrapped target

    In this project:
    - terminal positions use the final game reward
    - nonterminal positions use a bootstrapped estimate based on the next state's value

    Eligibility traces:
        trace <- gamma * lambda * trace + grad(V(S_t))

    This gives TD(lambda) its memory of earlier states, which lets the model
    assign credit backward through the episode more effectively than TD(0).

    Returns
    -------
    float
        Absolute TD error magnitude, useful for logging and graphs.
    """
    v_t, cache = model.forward_with_cache(x_t)
    delta = float(target) - float(v_t)

    # Manual NumPy gradients of V(S_t) with respect to all parameters.
    grads = model.gradients(cache)

    # Semi-gradient TD(lambda) parameter update.
    model.apply_td_update(
        traces=traces,
        grads=grads,
        delta=delta,
        alpha=alpha,
        gamma=gamma,
        lam=lam,
    )

    return float(abs(delta))