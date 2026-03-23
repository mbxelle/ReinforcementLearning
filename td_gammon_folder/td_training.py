"""
Main training loop for the TD-Gammon-style implementation.

This is the best file to explain the full RL pipeline because it connects:
- environment
- move selection
- TD target construction
- TD(lambda) updates
- evaluation
- CSV logging
"""

import csv
import random

from td_config import Config
from td_env import BackgammonEnv
from td_eval import evaluate_vs_random
from td_learning import td_lambda_update
from td_model import ValueNet
from td_search import choose_afterstate
from td_utils import epsilon_by_episode, set_all_seeds


def train_td_gammon(cfg: Config):
    """
    Run TD-Gammon-style self-play training.

    Pipeline:
    1. set seeds
    2. build the neural value network
    3. run self-play games
    4. after each turn, compute a TD target
    5. update the network with TD(lambda)
    6. periodically evaluate versus a random opponent
    7. save CSV files for graphing

    This is the central file that ties the whole project together.
    """
    set_all_seeds(cfg.seed)
    rng = random.Random(cfg.seed)

    # The hand-crafted board encoding uses 198 input features.
    input_dim = 198
    model = ValueNet(input_dim=input_dim, hidden_size=cfg.hidden_size)

    metrics_rows = []
    eval_rows = []

    print("Starting TD-Gammon-style self-play training...")
    print(f"Episodes: {cfg.episodes}")
    print(f"Network input size: {input_dim}")
    print(f"Hidden size: {cfg.hidden_size}")
    print(f"alpha={cfg.alpha}, lambda={cfg.lam}, gamma={cfg.gamma}")
    print(f"2-ply search enabled: {cfg.use_two_ply_search}")
    print()

    for episode in range(1, cfg.episodes + 1):
        env = BackgammonEnv()
        env.current_player = episode % 2  # alternate starting side for fairness
        epsilon = epsilon_by_episode(cfg, episode - 1)

        # Eligibility traces are reset at the start of each game.
        traces = model.zero_traces()

        td_errors = []
        move_counter = 0
        first_turn_log = None

        while env.winner() is None and env.turn_count < cfg.max_turns_per_game:
            player = env.current_player
            dice = env.roll_dice(rng)

            # Encode the current state from the current player's perspective.
            x_t = env.encode_for_player(player)

            # Choose one complete afterstate for the whole dice roll.
            chosen = choose_afterstate(
                model=model,
                env=env,
                player=player,
                dice=dice,
                epsilon=epsilon,
                rng=rng,
                use_two_ply_search=cfg.use_two_ply_search,
                two_ply_sample_rolls=cfg.two_ply_sample_rolls,
            )
            next_env = chosen["state"]

            winner = next_env.winner()

            if winner is not None:
                # Terminal target:
                # use the final game outcome reward from the perspective of the player
                # who just moved. This includes normal win / gammon / backgammon scoring.
                target = next_env.outcome_reward_for_player(player) / 3.0

            else:
                # Nonterminal bootstrapped target:
                # next_env.current_player is the opponent, so the network's value is
                # from the opponent's perspective. Because backgammon is zero-sum,
                # we negate it to convert back to the current player's perspective.
                v_next_opp = model.forward(next_env.encode_for_player(next_env.current_player))
                target = -float(v_next_opp)

            # TD(lambda) update:
            # move V(S_t) toward the target.
            abs_td = td_lambda_update(
                model=model,
                traces=traces,
                x_t=x_t,
                target=target,
                alpha=cfg.alpha,
                gamma=cfg.gamma,
                lam=cfg.lam,
            )

            td_errors.append(abs_td)
            move_counter += 1

            if first_turn_log is None:
                move_text = "; ".join(env.pretty_move(player, m) for m in chosen["moves"]) or "PASS"
                first_turn_log = (
                    f"dice={dice}, moves=[{move_text}], "
                    f"est_afterstate_value={chosen['estimated_value']:.3f}"
                )

            env = next_env

        winner = env.winner()

        # Emergency adjudication if turn limit is reached.
        if winner is None:
            p0 = env.pip_count(0)
            p1 = env.pip_count(1)
            winner = 0 if p0 < p1 else 1

        metrics_rows.append(
            {
                "episode": episode,
                "epsilon": round(epsilon, 6),
                "turns": env.turn_count,
                "network_updates": move_counter,
                "winner": winner,
                "winner_label": "P0" if winner == 0 else "P1",
                "p0_pip_count": env.pip_count(0),
                "p1_pip_count": env.pip_count(1),
                "avg_abs_td_error": sum(td_errors) / max(1, len(td_errors)),
            }
        )

        if episode == 1 or episode % cfg.eval_interval == 0:
            eval_stats = evaluate_vs_random(
                model=model,
                games=cfg.eval_games,
                seed=cfg.seed + episode,
                max_turns=cfg.max_turns_per_game,
                use_two_ply_search=cfg.use_two_ply_search,
                two_ply_sample_rolls=cfg.two_ply_sample_rolls,
            )
            eval_row = {"episode": episode, **eval_stats}
            eval_rows.append(eval_row)

            print(
                f"Episode {episode:5d} | turns={env.turn_count:3d} | "
                f"avg|TD error|={metrics_rows[-1]['avg_abs_td_error']:.4f} | "
                f"win_rate_vs_random={eval_stats['win_rate_vs_random']:.3f}"
            )

            if first_turn_log is not None:
                print(f"  Example first turn: {first_turn_log}")

    # Save CSV files for graphing.
    with open(cfg.metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "epsilon",
                "turns",
                "network_updates",
                "winner",
                "winner_label",
                "p0_pip_count",
                "p1_pip_count",
                "avg_abs_td_error",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    with open(cfg.eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "win_rate_vs_random",
                "loss_rate_vs_random",
                "avg_turns_eval",
            ],
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    print()
    print("Training complete.")
    print(f"Saved training metrics to: {cfg.metrics_csv}")
    print(f"Saved evaluation metrics to: {cfg.eval_csv}")
    print()
    print("Graph data:")
    print("  1) episode vs win_rate_vs_random")
    print("  2) episode vs avg_abs_td_error")
    print("  3) episode vs turns")
    print("  4) episode vs network_updates")

    return model, metrics_rows, eval_rows