import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

# ============================================================
# HEAVILY COMMENTED TD-GAMMON-STYLE DEMO
# ============================================================
# This is the same overall program structure as the user's file,
# but rewritten with many more comments so it is easier to study,
# present, and explain.
#
# Big picture:
# 1. We define a backgammon state.
# 2. We define an environment that knows the rules:
#    - how to roll dice
#    - how to find legal moves
#    - how to apply a move
#    - how to switch perspective between players
# 3. We define a tabular TD(lambda) agent that learns a value for states.
# 4. We train by self-play.
# 5. We evaluate against a random agent.
# 6. We print results and a short demo game.
#
# Important note:
# This is NOT the original neural-network TD-Gammon from Tesauro.
# This version is tabular, which makes it easier to understand,
# but much weaker than the real TD-Gammon.
# ============================================================

# A Move = (source, destination, die_used)
# Example: (23, 20, 3) means move a checker from point 24 to point 21 using die 3.
# We also allow special strings:
# - "bar" means a checker enters from the bar
# - "off" means a checker is borne off the board
Move = Tuple[object, object, int]

# An Action is a whole turn, which may contain multiple single moves
# because a player may need to use 2 dice (or 4 if doubles are rolled).
Action = Tuple[Move, ...]


@dataclass(frozen=True)
class BGState:
    """
    Stores the board from the CURRENT player's point of view.

    self_pts[i] = number of current player's checkers on point i
    opp_pts[i]  = number of opponent's checkers on point i

    self_bar / opp_bar = number of checkers on the bar
    self_off / opp_off = number of checkers already borne off

    Why use current-player perspective?
    ----------------------------------
    It makes the learning problem much cleaner. After a move, we can swap
    perspective so the next player always sees the board as "self".
    That means the same value function can be reused for both players.
    """
    self_pts: Tuple[int, ...]
    opp_pts: Tuple[int, ...]
    self_bar: int
    opp_bar: int
    self_off: int
    opp_off: int

    def key(self) -> Tuple:
        """
        Converts the state into a hashable key so it can be stored in a
        dictionary for tabular learning.
        """
        return (
            self.self_pts,
            self.opp_pts,
            self.self_bar,
            self.opp_bar,
            self.self_off,
            self.opp_off,
        )


class FullBackgammonEnv:
    """
    Environment that implements a simplified but fairly complete backgammon-like game.

    Responsibilities:
    - initialize the board
    - roll dice
    - detect terminal states
    - generate legal moves/actions
    - apply actions
    - compute shaping rewards
    - swap perspective after each turn
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Standard backgammon has 24 points.
        self.board_size = 24

        # Each player starts with 15 checkers.
        self.num_checkers = 15

        # Home board has 6 points.
        self.home_size = 6

    def initial_state(self) -> BGState:
        """
        Create the standard starting position, but represented from the
        current player's perspective.

        Index meaning:
        - point 0  means the farthest point from bearing off in this representation
        - point 23 means the closest point at the start side in this representation

        The current player moves from higher indices down toward 0, then off.
        """
        self_pts = [0] * 24
        opp_pts = [0] * 24

        # Standard backgammon opening arrangement for current player.
        self_pts[23] = 2
        self_pts[12] = 5
        self_pts[7] = 3
        self_pts[5] = 5

        # Opponent mirrored arrangement.
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
        """
        Roll two dice.

        If doubles are rolled, return four copies of that value, because in
        backgammon doubles are played four times.
        """
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        if d1 == d2:
            return [d1, d1, d1, d1]
        return [d1, d2]

    def is_terminal(self, state: BGState) -> bool:
        """
        The game ends when either player has borne off all 15 checkers.
        """
        return state.self_off == self.num_checkers or state.opp_off == self.num_checkers

    def all_in_home(self, state: BGState) -> bool:
        """
        Bearing off is only allowed when all of the current player's checkers
        are in the home board and none are on the bar.

        Since home board is points 0..5 in this perspective:
        - self_pts[6:] must all be zero
        - self_bar must be zero
        """
        if state.self_bar > 0:
            return False
        return sum(state.self_pts[self.home_size:]) == 0

    def swap_perspective(self, state: BGState) -> BGState:
        """
        After one player moves, we flip the board so the next player becomes
        the new "self".

        Key idea:
        - current opponent becomes new self
        - board points are reversed because direction of travel changes
        - bars and borne-off counts are swapped

        This is one of the most important design choices in the code because it
        allows one value function to be used for both players.
        """
        return BGState(
            self_pts=tuple(reversed(state.opp_pts)),
            opp_pts=tuple(reversed(state.self_pts)),
            self_bar=state.opp_bar,
            opp_bar=state.self_bar,
            self_off=state.opp_off,
            opp_off=state.self_off,
        )

    def point_open_for_self(self, state: BGState, dest: int) -> bool:
        """
        A destination point is legal if the opponent has at most 1 checker there.

        Backgammon rule:
        - 0 opponent checkers -> open
        - 1 opponent checker -> open, and that checker is hit
        - 2 or more opponent checkers -> blocked
        """
        return state.opp_pts[dest] <= 1

    def legal_single_die_moves(self, state: BGState, die: int) -> List[Move]:
        """
        Generate all legal moves using exactly one die.

        Move priority rule:
        If the player has checkers on the bar, they MUST re-enter from the bar first.
        So in that case, we only consider bar-entry moves.
        """
        moves: List[Move] = []

        # If any checker is on the bar, only bar-entry moves are allowed.
        if state.self_bar > 0:
            # Entering from the bar lands at board_size - die.
            # Example: die=1 lands on point 23, die=6 lands on point 18.
            dest = self.board_size - die
            if self.point_open_for_self(state, dest):
                moves.append(("bar", dest, die))
            return moves

        # Otherwise, try moving each checker already on the board.
        for src in range(self.board_size):
            if state.self_pts[src] == 0:
                continue  # no checker at this point, so nothing to move

            dest = src - die

            # Normal move that stays on the board.
            if dest >= 0:
                if self.point_open_for_self(state, dest):
                    moves.append((src, dest, die))

            # Possible bearing off move.
            else:
                # Cannot bear off unless all checkers are in home board.
                if not self.all_in_home(state):
                    continue

                # Exact bear-off: die takes checker exactly off the board.
                if dest == -1:
                    moves.append((src, "off", die))
                else:
                    # Oversized die can bear off only if there are no checkers on higher points.
                    # Example: checker on point 3 using die 6 can bear off only if all points above it are empty.
                    if sum(state.self_pts[src + 1:]) == 0:
                        moves.append((src, "off", die))

        return moves

    def apply_single_move(self, state: BGState, move: Move) -> Tuple[BGState, int, int]:
        """
        Apply one move and return:
        - the next state
        - whether a hit happened (0 or 1)
        - whether a bear-off happened (0 or 1)

        We track hit and bear-off events because they are used later in the
        shaping reward.
        """
        src, dest, _die = move

        # Convert tuples to lists so we can mutate them.
        self_pts = list(state.self_pts)
        opp_pts = list(state.opp_pts)
        self_bar = state.self_bar
        opp_bar = state.opp_bar
        self_off = state.self_off
        opp_off = state.opp_off

        hit_event = 0
        bearoff_event = 0

        # Remove checker from source.
        if src == "bar":
            self_bar -= 1
        else:
            self_pts[src] -= 1

        # Place checker at destination.
        if dest == "off":
            # Bearing off permanently removes checker from board.
            self_off += 1
            bearoff_event = 1
        else:
            # If opponent has a blot (exactly 1 checker), hit it to the bar.
            if opp_pts[dest] == 1:
                opp_pts[dest] = 0
                opp_bar += 1
                hit_event = 1

            # Put our checker on the destination point.
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
        """
        Generate all legal TURNS (not just single moves).

        Why is this more complicated than legal_single_die_moves?
        ---------------------------------------------------------
        Because a turn can contain multiple moves, and the order matters when
        the dice are different.

        Example:
        With dice [2, 5], you may need to consider:
        - play 2 then 5
        - play 5 then 2

        With doubles, we just use the repeated list once.

        Backgammon also has the rule that you must use as many dice as possible,
        so after generating all sequences we keep only the sequences with maximum length.
        """
        if len(dice) == 2 and dice[0] != dice[1]:
            orders = [dice, list(reversed(dice))]
        else:
            orders = [dice]

        all_sequences: List[Action] = []

        def recurse(cur_state: BGState, remaining_dice: List[int], seq: List[Move]):
            """
            Recursive helper that builds a full action sequence one die at a time.
            """
            if not remaining_dice:
                # Used all dice successfully.
                all_sequences.append(tuple(seq))
                return

            die = remaining_dice[0]
            moves = self.legal_single_die_moves(cur_state, die)

            # If no legal move exists for this die, this partial sequence ends here.
            if not moves:
                all_sequences.append(tuple(seq))
                return

            for mv in moves:
                nxt, _, _ = self.apply_single_move(cur_state, mv)
                recurse(nxt, remaining_dice[1:], seq + [mv])

        for order in orders:
            recurse(state, order, [])

        # Remove duplicates because different recursive paths / orders can lead to same sequence.
        unique_sequences = list(set(all_sequences))
        if not unique_sequences:
            return []

        # Backgammon rule: must use as many dice as possible.
        max_len = max(len(seq) for seq in unique_sequences)
        best = [seq for seq in unique_sequences if len(seq) == max_len]

        # Special rule for two different dice:
        # if only one die can be played, the larger die must be used.
        if len(dice) == 2 and dice[0] != dice[1] and max_len == 1:
            larger = max(dice)
            best = [seq for seq in best if seq[0][2] == larger]

        # Sort only for stable/reproducible ordering.
        best.sort(key=str)
        return best

    def pip_count(self, pts: Tuple[int, ...], bar: int) -> int:
        """
        Compute pip count.

        Pip count = total remaining distance to bear off.
        Lower pip count is generally better in racing positions.

        Each checker on point i contributes (i + 1) pips because point 0 is 1 step away.
        A checker on the bar is treated as distance 25.
        """
        total = 0
        for i, c in enumerate(pts):
            total += (i + 1) * c
        total += 25 * bar
        return total

    def blot_count(self, pts: Tuple[int, ...]) -> int:
        """
        Count blots (points with exactly one checker).

        Blots are vulnerable to being hit, so fewer blots is usually safer.
        """
        return sum(1 for c in pts if c == 1)

    def prime_strength(self, pts: Tuple[int, ...]) -> int:
        """
        Compute the length of the longest consecutive run of made points
        (points containing 2 or more checkers).

        Longer primes are generally stronger because they block movement.
        """
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
        Compute a small shaping reward for non-terminal moves.

        Why use shaping?
        ----------------
        If we only reward final wins/losses, learning is very slow because the
        feedback arrives at the end of a long game.

        So this function gives small intermediate rewards for progress such as:
        - reducing our pip count
        - increasing opponent bar count
        - bearing off checkers
        - reducing our blots
        - making stronger primes

        Important:
        The terminal win reward is still dominant. These shaping rewards are small.
        """
        before_pip_self = self.pip_count(before.self_pts, before.self_bar)
        before_pip_opp = self.pip_count(before.opp_pts, before.opp_bar)
        after_pip_self = self.pip_count(after_before_swap.self_pts, after_before_swap.self_bar)
        after_pip_opp = self.pip_count(after_before_swap.opp_pts, after_before_swap.opp_bar)

        # Reward reducing our own pip count more than helping opponent reduce theirs.
        pip_gain = (before_pip_self - after_pip_self) - 0.35 * (before_pip_opp - after_pip_opp)

        # Positive if our blot count decreases.
        blot_term = self.blot_count(before.self_pts) - self.blot_count(after_before_swap.self_pts)

        # Positive if our prime got stronger.
        prime_term = self.prime_strength(after_before_swap.self_pts) - self.prime_strength(before.self_pts)

        # Positive if opponent gets more checkers on bar, negative if we do.
        bar_term = (after_before_swap.opp_bar - before.opp_bar) - 0.5 * (after_before_swap.self_bar - before.self_bar)

        # Positive if we bear off more than opponent does.
        off_term = (after_before_swap.self_off - before.self_off) - 0.5 * (after_before_swap.opp_off - before.opp_off)

        return (
            0.0025 * pip_gain
            + 0.0200 * off_term
            + 0.0100 * bar_term
            + 0.0080 * blot_term
            + 0.0060 * prime_term
            + 0.0100 * hit_count
            + 0.0120 * bearoff_count
            - 0.0015  # small living penalty to discourage ultra-long games
        )

    def step(self, state: BGState, action: Action) -> Tuple[Optional[BGState], float, bool, int, int]:
        """
        Apply a whole turn (action) and return:
        - next_state (already perspective-swapped unless terminal)
        - reward
        - done
        - total hit count during turn
        - total bear-off count during turn

        Why swap perspective here?
        --------------------------
        Because after a player finishes a turn, it becomes the opponent's turn.
        By swapping before returning, the next state is already represented from
        the next player's point of view.
        """
        cur = state
        hit_count = 0
        bearoff_count = 0

        # Apply each single move inside the turn.
        for mv in action:
            cur, hit_event, bearoff_event = self.apply_single_move(cur, mv)
            hit_count += hit_event
            bearoff_count += bearoff_event

        # If current player finished all 15 checkers, game ends immediately.
        if cur.self_off == self.num_checkers:
            return None, 1.0, True, hit_count, bearoff_count

        # Otherwise give shaping reward and swap perspective for next turn.
        reward = self.shaped_progress(state, cur, hit_count, bearoff_count)
        return self.swap_perspective(cur), reward, False, hit_count, bearoff_count

    def pretty_move(self, move: Move) -> str:
        """
        Human-readable move text for printing demos.
        """
        src, dest, die = move
        src_txt = "BAR" if src == "bar" else str(src + 1)
        dest_txt = "OFF" if dest == "off" else str(dest + 1)
        return f"{src_txt}->{dest_txt} (die {die})"

    def compact_summary(self, state: BGState) -> str:
        """
        Short summary used in console output.
        """
        return (
            f"self_off={state.self_off}, opp_off={state.opp_off}, "
            f"self_bar={state.self_bar}, opp_bar={state.opp_bar}"
        )


class TDAgent:
    """
    Tabular TD(lambda) value-learning agent.

    What this agent learns:
    -----------------------
    It learns V(s), the estimated value of a state from the current player's perspective.

    High V(s)  -> good position for the current player
    Low V(s)   -> bad position for the current player

    Why TD(lambda)?
    ---------------
    TD(0) updates only the current state.
    TD(lambda) adds eligibility traces so recently visited states also receive credit/blame.
    This helps learning move information backward through a trajectory faster.
    """
    def __init__(self, alpha: float = 0.10, gamma: float = 0.99, lam: float = 0.70):
        # Base learning rate.
        self.alpha = alpha

        # Discount factor.
        self.gamma = gamma

        # Trace decay parameter lambda.
        self.lam = lam

        # Tabular value table: state_key -> value estimate.
        self.V: Dict[Tuple, float] = {}

        # Visit counts used to decay learning rate by state.
        self.visits: DefaultDict[Tuple, int] = defaultdict(int)

        # Eligibility traces: state_key -> current trace value.
        self.traces: DefaultDict[Tuple, float] = defaultdict(float)

        # Useful for reporting/debugging.
        self.last_td_error = 0.0

    def reset_episode(self):
        """
        Clear eligibility traces at the beginning of each new game.
        """
        self.traces.clear()

    def heuristic_value(self, state: BGState) -> float:
        """
        A handcrafted fallback estimate for unseen states.

        Why do this instead of defaulting to 0?
        --------------------------------------
        A heuristic gives the agent a stronger starting point before it has seen
        many states. That often makes early learning and action selection better.
        """
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

        # Clamp so heuristic stays in a reasonable range.
        return max(-0.95, min(0.95, val))

    def value(self, state: BGState) -> float:
        """
        Return learned value if present, otherwise heuristic fallback.
        """
        return self.V.get(state.key(), self.heuristic_value(state))

    def learning_rate(self, key: Tuple) -> float:
        """
        Decaying learning rate based on visit count.

        More visits -> smaller steps -> more stable estimates.
        Floor at 0.01 so updates do not vanish completely.
        """
        n = self.visits[key]
        return max(0.01, self.alpha / math.sqrt(max(1, n)))

    def update(self, state: BGState, next_state: Optional[BGState], reward: float, done: bool):
        """
        Perform one TD(lambda) update.

        Core idea:
        ---------
        We compare our current estimate V(s) to a target:

            target = r + gamma * (-V(s'))   if not terminal
            target = r                       if terminal

        Why negative V(next_state)?
        ---------------------------
        Because next_state is represented from the NEXT player's perspective.
        If the next player likes that state, that means it is bad for the current player.
        So from the current player's point of view, its value is the negative.

        Then TD error is:
            delta = target - V(s)

        With eligibility traces, we update not only s but also recent states.
        """
        s_key = state.key()
        old_v = self.V.get(s_key, self.heuristic_value(state))

        if done:
            target = reward
        else:
            target = reward + self.gamma * (-self.value(next_state))

        td_error = target - old_v
        self.last_td_error = td_error

        # Increase visit count for state s.
        self.visits[s_key] += 1

        # Replace/accumulate trace for the current state.
        self.traces[s_key] += 1.0

        keys_to_delete = []

        # Update every state currently carrying some eligibility trace.
        for key in list(self.traces.keys()):
            # If state has never had an explicit table entry, start at 0.
            # (Original file used 0.0 here too.)
            base = self.V.get(key)
            if base is None:
                base = 0.0

            lr = self.learning_rate(key)

            # TD(lambda) update:
            # V(k) <- V(k) + alpha * delta * eligibility(k)
            self.V[key] = base + lr * td_error * self.traces[key]

            # Decay the trace for future steps.
            self.traces[key] *= self.gamma * self.lam

            # Remove tiny traces to keep dictionary small.
            if self.traces[key] < 1e-5:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.traces[key]


def choose_greedy_action(env: FullBackgammonEnv, agent: TDAgent, state: BGState, legal_actions: List[Action]) -> Action:
    """
    Choose the action with the best one-step lookahead score.

    For each legal action we compute:
        immediate reward + discounted value of next state

    Again we negate agent.value(next_state) because next_state is from the next player's perspective.
    """
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
    """
    Epsilon-greedy exploration.

    With probability epsilon:
        choose random legal action
    Otherwise:
        choose greedy best action

    This allows the agent to explore during training.
    """
    if env.rng.random() < epsilon:
        return env.rng.choice(legal_actions)
    return choose_greedy_action(env, agent, state, legal_actions)



def quick_sanity_check():
    """
    Small test to make sure the environment is producing dice and legal actions.
    Good for debugging before running long experiments.
    """
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
    Train the agent for some number of self-play episodes.

    Training improvements used here:
    - epsilon decays from epsilon_start to epsilon_end
    - TD(lambda) updates
    - mixed opponent policy:
        sometimes greedy wrt current value function
        sometimes random
    - unfinished games get penalized so learning does not drift toward endless play

    mixed_opponent_prob:
        probability that opponent uses greedy action instead of random
    """
    if episodes <= 0:
        return

    for ep in range(episodes):
        # Linearly decay epsilon across this training chunk.
        frac = ep / max(1, episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        state = env.initial_state()
        turns = 0
        agent_turn = True  # alternate between agent and opponent
        agent.reset_episode()

        while True:
            turns += 1

            # If game goes too long, force an ending with a penalty.
            if turns > max_turns_per_game:
                agent.update(state, None, -0.20, True)
                break

            dice = env.roll_dice()
            legal_actions = env.generate_legal_actions(state, dice)

            if not legal_actions:
                # No move possible: effectively pass turn by swapping perspective.
                next_state = env.swap_perspective(state)
                reward = -0.001
                done = False
            else:
                if agent_turn:
                    action = choose_training_action(env, agent, state, legal_actions, epsilon)
                else:
                    # Opponent is sometimes greedy and sometimes random.
                    if env.rng.random() < mixed_opponent_prob:
                        action = choose_greedy_action(env, agent, state, legal_actions)
                    else:
                        action = env.rng.choice(legal_actions)
                next_state, reward, done, _, _ = env.step(state, action)

            if agent_turn:
                # Standard update from agent's current perspective.
                agent.update(state, next_state, reward, done)
            else:
                # Here state is from the opponent's perspective because of perspective swapping.
                # So we convert reward to what it means for the learning function consistently.
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
    """
    Evaluate greedy agent against a random opponent.

    Metrics returned:
    - win rate
    - average game length
    - number of completed / unfinished games
    - average borne-off counts
    - average bar counts
    - average hit and bear-off events
    - average absolute TD error during agent turns
    - number of known states in the table
    """
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
                # Compute one-step TD error only for reporting.
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

                # For completed games we record who finished all 15 checkers.
                if agent_turn:
                    total_self_off += env.num_checkers
                    total_opp_off += 0
                else:
                    total_self_off += 0
                    total_opp_off += env.num_checkers

                # These are set to 0 here because terminal aggregation is simplified in this script.
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
    """
    Train incrementally and evaluate at multiple checkpoints.

    Example:
    checkpoints = [0, 25, 50, 100]

    Means:
    - evaluate with no training
    - train to 25 episodes, evaluate
    - train to 50 episodes, evaluate
    - train to 100 episodes, evaluate
    """
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

        # Estimated value of the opening position.
        start_value = agent.value(env.initial_state())

        # Evaluate current policy.
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

    # Save results so they can be plotted later or used in report slides.
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
    """
    Play one printed greedy-vs-greedy style demo from the learned value function.

    This is useful for showing a sample trajectory in a presentation.
    """
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
    # 1. Quick environment sanity check.
    quick_sanity_check()

    # 2. Training checkpoints.
    checkpoints = [0, 25, 50, 100, 250, 500]

    # 3. Run training + evaluation.
    rows, env, agent = run_experiment(
        checkpoints=checkpoints,
        alpha=0.10,
        gamma=0.99,
        lam=0.70,
        seed=42,
        eval_games=50,
        csv_filename="td_gammon_full_tabular_improved_stats.csv",
    )

    # 4. Show one demo game after training.
    play_demo_game(env=env, agent=agent, seed=999, max_turns=300)
