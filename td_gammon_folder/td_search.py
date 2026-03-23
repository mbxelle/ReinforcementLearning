"""
Move selection and afterstate search code.

This file is split out because:
- it is one of the most important TD-Gammon ideas
- it is easier to explain when isolated from training code

Core idea:
Generate legal afterstates, score them with the value network,
and choose the best one.
"""

import random
from typing import Dict, Tuple

from td_env import BackgammonEnv
from td_model import ValueNet


def value_of_state(model: ValueNet, env: BackgammonEnv, player: int) -> float:
    """
    Evaluate one state from a specified player's perspective.

    Parameters
    ----------
    model:
        The neural value network.
    env:
        The board position to evaluate.
    player:
        The player whose perspective we want.

    Returns
    -------
    float
        Estimated value in [-1, +1].
    """
    x = env.encode_for_player(player)
    return float(model.forward(x))


def expected_value_two_ply(
    model: ValueNet,
    afterstate: BackgammonEnv,
    root_player: int,
    rng: random.Random,
    sample_rolls: int,
) -> float:
    """
    Estimate a 2-ply value for an afterstate.

    Meaning:
    - root player already moved
    - now it is opponent's turn
    - we approximate the opponent's response over sampled dice rolls

    Why this helps:
    Plain 1-ply afterstate evaluation scores the position immediately after
    the current player's move.
    2-ply goes one step deeper by considering likely opponent replies.

    This is more expensive, but often produces stronger move selection.

    Important perspective detail:
    afterstate.current_player is the opponent.
    The network evaluates from the side to move.
    So when converting back to the root player's perspective, we negate.
    """
    opp = afterstate.current_player
    total_value = 0.0

    for _ in range(sample_rolls):
        dice = afterstate.roll_dice(rng)
        opp_candidates = afterstate.legal_turn_afterstates(opp, dice)

        if not opp_candidates:
            # If somehow no candidate exists, just score the current afterstate.
            opp_value = value_of_state(model, afterstate, opp)
            root_value = -opp_value
            total_value += root_value
            continue

        # Opponent acts greedily from their own perspective.
        best_opp_value = -1e9
        for cand in opp_candidates:
            v = value_of_state(model, cand["state"], cand["state"].current_player)
            if v > best_opp_value:
                best_opp_value = v

        # Convert opponent perspective back to root player's perspective.
        total_value += -best_opp_value

    return total_value / max(1, sample_rolls)


def choose_afterstate(
    model: ValueNet,
    env: BackgammonEnv,
    player: int,
    dice: Tuple[int, int],
    epsilon: float,
    rng: random.Random,
    use_two_ply_search: bool = False,
    two_ply_sample_rolls: int = 8,
) -> Dict:
    """
    Select one legal afterstate using epsilon-greedy move selection.

    CORE TD-GAMMON IDEA:
    Instead of learning Q(s,a), this code evaluates AFTERSTATES.

    AFTERSTATE = the board after a full turn has been completed.

    Main steps:
    1. generate all legal full-turn afterstates
    2. score each afterstate with the neural network
    3. choose the best one, or a random one with probability epsilon

    Perspective detail:
    The afterstate's current_player is the opponent, so the raw neural-network
    value is from the opponent's perspective. Since the game is zero-sum, the
    current player's value is the negative of that value.

    If use_two_ply_search is True, we do a shallow lookahead by estimating the
    opponent's reply quality before choosing the move.
    """
    candidates = env.legal_turn_afterstates(player, dice)

    # Random exploration during training.
    if rng.random() < epsilon:
        choice = rng.choice(candidates)
        choice = dict(choice)

        if use_two_ply_search:
            est = expected_value_two_ply(
                model=model,
                afterstate=choice["state"],
                root_player=player,
                rng=rng,
                sample_rolls=two_ply_sample_rolls,
            )
        else:
            choice_player = choice["state"].current_player
            est = -value_of_state(model, choice["state"], choice_player)

        choice["estimated_value"] = est
        return choice

    best = None
    best_val = -1e9

    for cand in candidates:
        if use_two_ply_search:
            cur_value = expected_value_two_ply(
                model=model,
                afterstate=cand["state"],
                root_player=player,
                rng=rng,
                sample_rolls=two_ply_sample_rolls,
            )
        else:
            # cand["state"].current_player is the opponent because this is an afterstate.
            opp_value = value_of_state(model, cand["state"], cand["state"].current_player)
            cur_value = -opp_value

        if cur_value > best_val:
            best_val = cur_value
            best = cand

    best = dict(best)
    best["estimated_value"] = best_val
    return best