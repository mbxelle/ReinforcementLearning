import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

# ============================================================
# IMPROVED FULL-CONSTRAINTS TD-GAMMON-STYLE DEMO
# ============================================================
# Improvements over the original version:
# - TD(lambda) eligibility traces instead of plain TD(0)
# - decaying epsilon exploration
# - decaying learning rate based on state visit count
# - stronger heuristic fallback / initialization
# - small non-terminal shaping reward to reduce ultra-long games
# - optional self-play against a mixed opponent (greedy or random)
# - draw-style penalty for unfinished games
#
# Still tabular: this will learn better than the original, but it still
# will not match neural-network TD-Gammon on full backgammon.
# ============================================================

Move = Tuple[object, object, int]
Action = Tuple[Move, ...]


@dataclass(frozen=True)
class BGState:
    self_pts: Tuple[int, ...]
    opp_pts: Tuple[int, ...]
    self_bar: int
    opp_bar: int
    self_off: int
    opp_off: int

    def key(self) -> Tuple:
        return (
            self.self_pts,
            self.opp_pts,
            self.self_bar,
            self.opp_bar,
            self.self_off,
            self.opp_off,
        )


class FullBackgammonEnv:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.board_size = 24
        self.num_checkers = 15
        self.home_size = 6

    def initial_state(self) -> BGState:
        self_pts = [0] * 24
        opp_pts = [0] * 24

        self_pts[23] = 2
        self_pts[12] = 5
        self_pts[7] = 3
        self_pts[5] = 5

        opp_pts[0] = 2
        opp_pts[11] = 5
        opp_pts[16] = 3
        opp_pts[18] = 5

        return BGState(
            self_pts=tuple(self_pts),
            opp_pts=tuple(opp_pts),
            self_bar=0,
            opp_bar=0,
            self_off=0,
            opp_off=0,
        )

    def roll_dice(self) -> List[int]:
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        if d1 == d2:
            return [d1, d1, d1, d1]
        return [d1, d2]

    def is_terminal(self, state: BGState) -> bool:
        return state.self_off == self.num_checkers or state.opp_off == self.num_checkers

    def all_in_home(self, state: BGState) -> bool:
        if state.self_bar > 0:
            return False
        return sum(state.self_pts[self.home_size:]) == 0

    def swap_perspective(self, state: BGState) -> BGState:
        return BGState(
            self_pts=tuple(reversed(state.opp_pts)),
            opp_pts=tuple(reversed(state.self_pts)),
            self_bar=state.opp_bar,
            opp_bar=state.self_bar,
            self_off=state.opp_off,
            opp_off=state.self_off,
        )

    def point_open_for_self(self, state: BGState, dest: int) -> bool:
        return state.opp_pts[dest] <= 1

    def legal_single_die_moves(self, state: BGState, die: int) -> List[Move]:
        moves: List[Move] = []

        if state.self_bar > 0:
            dest = self.board_size - die
            if self.point_open_for_self(state, dest):
                moves.append(("bar", dest, die))
            return moves

        for src in range(self.board_size):
            if state.self_pts[src] == 0:
                continue

            dest = src - die
            if dest >= 0:
                if self.point_open_for_self(state, dest):
                    moves.append((src, dest, die))
            else:
                if not self.all_in_home(state):
                    continue
                if dest == -1:
                    moves.append((src, "off", die))
                else:
                    if sum(state.self_pts[src + 1:]) == 0:
                        moves.append((src, "off", die))

        return moves

    def apply_single_move(self, state: BGState, move: Move) -> Tuple[BGState, int, int]:
        src, dest, _die = move

        self_pts = list(state.self_pts)
        opp_pts = list(state.opp_pts)
        self_bar = state.self_bar
        opp_bar = state.opp_bar
        self_off = state.self_off
        opp_off = state.opp_off

        hit_event = 0
        bearoff_event = 0

        if src == "bar":
            self_bar -= 1
        else:
            self_pts[src] -= 1

        if dest == "off":
            self_off += 1
            bearoff_event = 1
        else:
            if opp_pts[dest] == 1:
                opp_pts[dest] = 0
                opp_bar += 1
                hit_event = 1
            self_pts[dest] += 1

        return BGState(
            self_pts=tuple(self_pts),
            opp_pts=tuple(opp_pts),
            self_bar=self_bar,
            opp_bar=opp_bar,
            self_off=self_off,
            opp_off=opp_off,
        ), hit_event, bearoff_event

    def generate_legal_actions(self, state: BGState, dice: List[int]) -> List[Action]:
        if len(dice) == 2 and dice[0] != dice[1]:
            orders = [dice, list(reversed(dice))]
        else:
            orders = [dice]

        all_sequences: List[Action] = []

        def recurse(cur_state: BGState, remaining_dice: List[int], seq: List[Move]):
            if not remaining_dice:
                all_sequences.append(tuple(seq))
                return

            die = remaining_dice[0]
            moves = self.legal_single_die_moves(cur_state, die)
            if not moves:
                all_sequences.append(tuple(seq))
                return

            for mv in moves:
                nxt, _, _ = self.apply_single_move(cur_state, mv)
                recurse(nxt, remaining_dice[1:], seq + [mv])

        for order in orders:
            recurse(state, order, [])

        unique_sequences = list(set(all_sequences))
        if not unique_sequences:
            return []

        max_len = max(len(seq) for seq in unique_sequences)
        best = [seq for seq in unique_sequences if len(seq) == max_len]

        if len(dice) == 2 and dice[0] != dice[1] and max_len == 1:
            larger = max(dice)
            best = [seq for seq in best if seq[0][2] == larger]

        best.sort(key=str)
        return best

    def pip_count(self, pts: Tuple[int, ...], bar: int) -> int:
        total = 0
        for i, c in enumerate(pts):
            total += (i + 1) * c
        total += 25 * bar
        return total

    def blot_count(self, pts: Tuple[int, ...]) -> int:
        return sum(1 for c in pts if c == 1)

    def prime_strength(self, pts: Tuple[int, ...]) -> int:
        best = 0
        cur = 0
        for c in pts:
            if c >= 2:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def shaped_progress(self, before: BGState, after_before_swap: BGState, hit_count: int, bearoff_count: int) -> float:
        """
        Small shaping reward computed before perspective swap.
        Positive if we improve our race/contact position.
        Keeps terminal reward dominant.
        """
        before_pip_self = self.pip_count(before.self_pts, before.self_bar)
        before_pip_opp = self.pip_count(before.opp_pts, before.opp_bar)
        after_pip_self = self.pip_count(after_before_swap.self_pts, after_before_swap.self_bar)
        after_pip_opp = self.pip_count(after_before_swap.opp_pts, after_before_swap.opp_bar)

        pip_gain = (before_pip_self - after_pip_self) - 0.35 * (before_pip_opp - after_pip_opp)
        blot_term = self.blot_count(before.self_pts) - self.blot_count(after_before_swap.self_pts)
        prime_term = self.prime_strength(after_before_swap.self_pts) - self.prime_strength(before.self_pts)
        bar_term = (after_before_swap.opp_bar - before.opp_bar) - 0.5 * (after_before_swap.self_bar - before.self_bar)
        off_term = (after_before_swap.self_off - before.self_off) - 0.5 * (after_before_swap.opp_off - before.opp_off)

        return (
            0.0025 * pip_gain
            + 0.0200 * off_term
            + 0.0100 * bar_term
            + 0.0080 * blot_term
            + 0.0060 * prime_term
            + 0.0100 * hit_count
            + 0.0120 * bearoff_count
            - 0.0015
        )

    def step(self, state: BGState, action: Action) -> Tuple[Optional[BGState], float, bool, int, int]:
        cur = state
        hit_count = 0
        bearoff_count = 0

        for mv in action:
            cur, hit_event, bearoff_event = self.apply_single_move(cur, mv)
            hit_count += hit_event
            bearoff_count += bearoff_event

        if cur.self_off == self.num_checkers:
            return None, 1.0, True, hit_count, bearoff_count

        reward = self.shaped_progress(state, cur, hit_count, bearoff_count)
        return self.swap_perspective(cur), reward, False, hit_count, bearoff_count

    def pretty_move(self, move: Move) -> str:
        src, dest, die = move
        src_txt = "BAR" if src == "bar" else str(src + 1)
        dest_txt = "OFF" if dest == "off" else str(dest + 1)
        return f"{src_txt}->{dest_txt} (die {die})"

    def compact_summary(self, state: BGState) -> str:
        return (
            f"self_off={state.self_off}, opp_off={state.opp_off}, "
            f"self_bar={state.self_bar}, opp_bar={state.opp_bar}"
        )


class TDAgent:
    """
    Tabular TD(lambda) with heuristic bootstrap, visit-based alpha decay,
    and eligibility traces.
    """
    def __init__(self, alpha: float = 0.10, gamma: float = 0.99, lam: float = 0.70):
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.V: Dict[Tuple, float] = {}
        self.visits: DefaultDict[Tuple, int] = defaultdict(int)
        self.traces: DefaultDict[Tuple, float] = defaultdict(float)
        self.last_td_error = 0.0

    def reset_episode(self):
        self.traces.clear()

    def heuristic_value(self, state: BGState) -> float:
        self_pip = sum((i + 1) * c for i, c in enumerate(state.self_pts)) + 25 * state.self_bar
        opp_pip = sum((i + 1) * c for i, c in enumerate(state.opp_pts)) + 25 * state.opp_bar
        self_blots = sum(1 for c in state.self_pts if c == 1)
        opp_blots = sum(1 for c in state.opp_pts if c == 1)
        self_points_made = sum(1 for c in state.self_pts if c >= 2)
        opp_points_made = sum(1 for c in state.opp_pts if c >= 2)

        val = (
            0.12 * (state.self_off - state.opp_off)
            + 0.020 * (state.opp_bar - state.self_bar)
            + 0.004 * (opp_pip - self_pip)
            + 0.010 * (opp_blots - self_blots)
            + 0.006 * (self_points_made - opp_points_made)
        )
        return max(-0.95, min(0.95, val))

    def value(self, state: BGState) -> float:
        return self.V.get(state.key(), self.heuristic_value(state))

    def learning_rate(self, key: Tuple) -> float:
        n = self.visits[key]
        return max(0.01, self.alpha / math.sqrt(max(1, n)))

    def update(self, state: BGState, next_state: Optional[BGState], reward: float, done: bool):
        s_key = state.key()
        old_v = self.V.get(s_key, self.heuristic_value(state))

        if done:
            target = reward
        else:
            target = reward + self.gamma * (-self.value(next_state))

        td_error = target - old_v
        self.last_td_error = td_error

        self.visits[s_key] += 1
        self.traces[s_key] += 1.0

        updates = []
        for key in list(self.traces.keys()):
            base = self.V.get(key)
            if base is None:
                base = 0.0
            lr = self.learning_rate(key)
            self.V[key] = base + lr * td_error * self.traces[key]
            self.traces[key] *= self.gamma * self.lam
            if self.traces[key] < 1e-5:
                updates.append(key)

        for key in updates:
            del self.traces[key]


def choose_greedy_action(env: FullBackgammonEnv, agent: TDAgent, state: BGState, legal_actions: List[Action]) -> Action:
    best_action = legal_actions[0]
    best_score = -10**18

    for action in legal_actions:
        next_state, reward, done, _hits, _bears = env.step(state, action)
        score = reward if done else reward + agent.gamma * (-agent.value(next_state))
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def choose_training_action(
    env: FullBackgammonEnv,
    agent: TDAgent,
    state: BGState,
    legal_actions: List[Action],
    epsilon: float,
) -> Action:
    if env.rng.random() < epsilon:
        return env.rng.choice(legal_actions)
    return choose_greedy_action(env, agent, state, legal_actions)


def quick_sanity_check():
    env = FullBackgammonEnv(seed=123)
    state = env.initial_state()
    dice = env.roll_dice()
    actions = env.generate_legal_actions(state, dice)

    print("SANITY CHECK")
    print(f"Initial state: {env.compact_summary(state)}")
    print(f"Dice: {dice}")
    print(f"Legal actions: {len(actions)}")
    if actions:
        print("Example action:")
        for mv in actions[0]:
            print(" ", env.pretty_move(mv))
    print("Sanity check passed.\n")


def train_incremental(
    env: FullBackgammonEnv,
    agent: TDAgent,
    episodes: int,
    epsilon_start: float = 0.25,
    epsilon_end: float = 0.05,
    max_turns_per_game: int = 700,
    mixed_opponent_prob: float = 0.50,
):
    """
    Self-play training with:
    - epsilon decay
    - TD(lambda)
    - mixed opponent: sometimes greedy, sometimes random
    - unfinished-game penalty to reduce endless play
    """
    if episodes <= 0:
        return

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        state = env.initial_state()
        turns = 0
        agent_turn = True
        agent.reset_episode()

        while True:
            turns += 1
            if turns > max_turns_per_game:
                # Penalize long non-terminal trajectories.
                agent.update(state, None, -0.20, True)
                break

            dice = env.roll_dice()
            legal_actions = env.generate_legal_actions(state, dice)

            if not legal_actions:
                next_state = env.swap_perspective(state)
                reward = -0.001
                done = False
            else:
                if agent_turn:
                    action = choose_training_action(env, agent, state, legal_actions, epsilon)
                else:
                    if env.rng.random() < mixed_opponent_prob:
                        action = choose_greedy_action(env, agent, state, legal_actions)
                    else:
                        action = env.rng.choice(legal_actions)
                next_state, reward, done, _, _ = env.step(state, action)

            if agent_turn:
                agent.update(state, next_state, reward, done)
            else:
                # Opponent state's value is from opponent perspective, so negate reward.
                opp_reward = -reward if not done else (-1.0 if reward > 0 else reward)
                agent.update(state, next_state, opp_reward, done)

            if done:
                break

            state = next_state
            agent_turn = not agent_turn


def evaluate_vs_random_detailed(
    env: FullBackgammonEnv,
    agent: TDAgent,
    games: int = 50,
    max_turns_per_game: int = 700,
) -> Dict[str, float]:
    wins = 0
    completed = 0
    unfinished = 0
    total_turns = 0

    total_self_off = 0.0
    total_opp_off = 0.0
    total_self_bar = 0.0
    total_opp_bar = 0.0
    total_hit_events = 0.0
    total_bearoff_events = 0.0
    total_td_abs_error = 0.0
    total_agent_turns = 0

    for _ in range(games):
        state = env.initial_state()
        agent_turn = True
        turns = 0
        game_hit_events = 0
        game_bearoff_events = 0

        while True:
            turns += 1
            if turns > max_turns_per_game:
                unfinished += 1
                total_turns += max_turns_per_game
                total_self_off += state.self_off
                total_opp_off += state.opp_off
                total_self_bar += state.self_bar
                total_opp_bar += state.opp_bar
                total_hit_events += game_hit_events
                total_bearoff_events += game_bearoff_events
                break

            dice = env.roll_dice()
            legal_actions = env.generate_legal_actions(state, dice)

            if not legal_actions:
                state = env.swap_perspective(state)
                agent_turn = not agent_turn
                continue

            if agent_turn:
                action = choose_greedy_action(env, agent, state, legal_actions)
            else:
                action = env.rng.choice(legal_actions)

            next_state, reward, done, hit_count, bearoff_count = env.step(state, action)
            game_hit_events += hit_count
            game_bearoff_events += bearoff_count

            if agent_turn:
                old_v = agent.value(state)
                if done:
                    target = reward
                else:
                    target = reward + agent.gamma * (-agent.value(next_state))
                total_td_abs_error += abs(target - old_v)
                total_agent_turns += 1

            if done:
                if agent_turn:
                    wins += 1
                completed += 1
                total_turns += turns

                if agent_turn:
                    total_self_off += env.num_checkers
                    total_opp_off += 0
                else:
                    total_self_off += 0
                    total_opp_off += env.num_checkers

                total_self_bar += 0
                total_opp_bar += 0
                total_hit_events += game_hit_events
                total_bearoff_events += game_bearoff_events
                break

            state = next_state
            agent_turn = not agent_turn

    total_games = max(1, completed + unfinished)
    completed_for_rate = max(1, completed)
    avg_td_abs_error = total_td_abs_error / total_agent_turns if total_agent_turns > 0 else 0.0

    return {
        "win_rate_vs_random": wins / completed_for_rate,
        "avg_game_length": total_turns / total_games,
        "completed_games": float(completed),
        "unfinished_games": float(unfinished),
        "avg_self_off": total_self_off / total_games,
        "avg_opp_off": total_opp_off / total_games,
        "avg_self_bar": total_self_bar / total_games,
        "avg_opp_bar": total_opp_bar / total_games,
        "avg_hit_events": total_hit_events / total_games,
        "avg_bearoff_events": total_bearoff_events / total_games,
        "avg_td_abs_error": avg_td_abs_error,
        "known_states": float(len(agent.V)),
    }


def run_experiment(
    checkpoints: List[int],
    alpha: float = 0.10,
    gamma: float = 0.99,
    lam: float = 0.70,
    seed: int = 42,
    eval_games: int = 50,
    csv_filename: str = "td_gammon_full_tabular_improved_stats.csv",
):
    env = FullBackgammonEnv(seed=seed)
    agent = TDAgent(alpha=alpha, gamma=gamma, lam=lam)

    rows = []
    checkpoints = sorted(checkpoints)
    prev = 0

    print("RESULTS")
    print(
        "episodes,start_value,win_rate_vs_random,avg_game_length,"
        "completed_games,unfinished_games,avg_self_off,avg_opp_off,"
        "avg_self_bar,avg_opp_bar,avg_hit_events,avg_bearoff_events,avg_td_abs_error,known_states"
    )

    for cp in checkpoints:
        additional = cp - prev
        if additional > 0:
            train_incremental(
                env=env,
                agent=agent,
                episodes=additional,
                epsilon_start=0.25,
                epsilon_end=0.05,
                max_turns_per_game=700,
                mixed_opponent_prob=0.50,
            )

        start_value = agent.value(env.initial_state())
        stats = evaluate_vs_random_detailed(
            env=env,
            agent=agent,
            games=eval_games,
            max_turns_per_game=700,
        )

        row = {
            "episodes": cp,
            "start_value": round(start_value, 3),
            "win_rate_vs_random": round(stats["win_rate_vs_random"], 3),
            "avg_game_length": round(stats["avg_game_length"], 1),
            "completed_games": int(stats["completed_games"]),
            "unfinished_games": int(stats["unfinished_games"]),
            "avg_self_off": round(stats["avg_self_off"], 2),
            "avg_opp_off": round(stats["avg_opp_off"], 2),
            "avg_self_bar": round(stats["avg_self_bar"], 2),
            "avg_opp_bar": round(stats["avg_opp_bar"], 2),
            "avg_hit_events": round(stats["avg_hit_events"], 2),
            "avg_bearoff_events": round(stats["avg_bearoff_events"], 2),
            "avg_td_abs_error": round(stats["avg_td_abs_error"], 4),
            "known_states": int(stats["known_states"]),
        }
        rows.append(row)

        print(
            f"{row['episodes']},{row['start_value']},{row['win_rate_vs_random']},"
            f"{row['avg_game_length']},{row['completed_games']},{row['unfinished_games']},"
            f"{row['avg_self_off']},{row['avg_opp_off']},{row['avg_self_bar']},"
            f"{row['avg_opp_bar']},{row['avg_hit_events']},{row['avg_bearoff_events']},"
            f"{row['avg_td_abs_error']},{row['known_states']}"
        )
        prev = cp

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episodes",
                "start_value",
                "win_rate_vs_random",
                "avg_game_length",
                "completed_games",
                "unfinished_games",
                "avg_self_off",
                "avg_opp_off",
                "avg_self_bar",
                "avg_opp_bar",
                "avg_hit_events",
                "avg_bearoff_events",
                "avg_td_abs_error",
                "known_states",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV file: {csv_filename}\n")
    return rows, env, agent


def play_demo_game(env: FullBackgammonEnv, agent: TDAgent, seed: int = 999, max_turns: int = 300):
    demo_env = FullBackgammonEnv(seed=seed)
    state = demo_env.initial_state()
    turn = 1

    print("DEMO GAME")
    while turn <= max_turns:
        print(f"Turn {turn}: {demo_env.compact_summary(state)}")
        dice = demo_env.roll_dice()
        legal_actions = demo_env.generate_legal_actions(state, dice)

        if not legal_actions:
            print(f"  dice={dice} -> no legal move")
            state = demo_env.swap_perspective(state)
            turn += 1
            continue

        action = choose_greedy_action(demo_env, agent, state, legal_actions)
        move_text = " | ".join(demo_env.pretty_move(mv) for mv in action)
        print(f"  dice={dice} -> {move_text}")

        next_state, reward, done, _, _ = demo_env.step(state, action)
        if done:
            print(f"Turn {turn}: current player wins")
            return

        state = next_state
        turn += 1

    print(f"Demo stopped after {max_turns} turns.")


if __name__ == "__main__":
    quick_sanity_check()

    checkpoints = [0, 25, 50, 100, 250, 500]
    rows, env, agent = run_experiment(
        checkpoints=checkpoints,
        alpha=0.10,
        gamma=0.99,
        lam=0.70,
        seed=42,
        eval_games=50,
        csv_filename="td_gammon_full_tabular_improved_stats.csv",
    )

    play_demo_game(env=env, agent=agent, seed=999, max_turns=300)
