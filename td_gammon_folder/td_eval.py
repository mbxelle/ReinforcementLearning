"""
Evaluation code for the TD-Gammon-style system.

Training is done by self-play.
Evaluation is done against a random opponent.

Keeping this separate makes it easier to explain the difference between:
- learning signal
- performance measurement
"""

from typing import Dict

import random

from td_env import BackgammonEnv
from td_model import ValueNet
from td_search import choose_afterstate


def evaluate_vs_random(
    model: ValueNet,
    games: int,
    seed: int,
    max_turns: int,
    use_two_ply_search: bool = False,
    two_ply_sample_rolls: int = 8,
) -> Dict[str, float]:
    """
    Evaluate the learned agent against a random-move opponent.

    Why this matters:
    The model trains by self-play, but to measure progress we want an
    external benchmark. A random opponent is simple, fast, and gives a
    clear win-rate signal for graphs.

    To reduce side bias:
    - the learned agent alternates between player 0 and player 1
    - the starting side also alternates

    Returns
    -------
    dict
        Contains:
        - win_rate_vs_random
        - loss_rate_vs_random
        - avg_turns_eval
    """
    rng = random.Random(seed)

    wins = 0
    losses = 0
    total_turns = 0

    for g in range(games):
        env = BackgammonEnv()
        learned_player = g % 2
        env.current_player = g % 2

        while env.winner() is None and env.turn_count < max_turns:
            player = env.current_player
            dice = env.roll_dice(rng)

            if player == learned_player:
                chosen = choose_afterstate(
                    model=model,
                    env=env,
                    player=player,
                    dice=dice,
                    epsilon=0.0,
                    rng=rng,
                    use_two_ply_search=use_two_ply_search,
                    two_ply_sample_rolls=two_ply_sample_rolls,
                )
                env = chosen["state"]
            else:
                candidates = env.legal_turn_afterstates(player, dice)
                env = rng.choice(candidates)["state"]

        winner = env.winner()

        # Emergency adjudication if a very long game hits the turn limit.
        if winner is None:
            p0 = env.pip_count(0)
            p1 = env.pip_count(1)
            winner = 0 if p0 < p1 else 1

        total_turns += env.turn_count

        if winner == learned_player:
            wins += 1
        else:
            losses += 1

    return {
        "win_rate_vs_random": wins / max(1, games),
        "loss_rate_vs_random": losses / max(1, games),
        "avg_turns_eval": total_turns / max(1, games),
    }