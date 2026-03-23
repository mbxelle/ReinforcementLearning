"""
Backgammon environment for the TD-Gammon-style implementation.

This file contains:
- full-board representation
- dice rolling
- legal single-die move generation
- full-turn move generation
- bar, hitting, bearing off
- feature encoding for the neural network
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class BackgammonEnv:
    """
    Full-board backgammon environment with the main game mechanics needed for a
    TD-Gammon style demo.

    Board indexing:
      - We store points as integers 0..23.
      - Player 0 moves from high index to low index: 23 -> 0 -> OFF
      - Player 1 moves from low index to high index: 0 -> 23 -> OFF

    Representation:
      points[player][point] = number of that player's checkers on that point
      bar[player]           = number of that player's checkers on the bar
      off[player]           = number of that player's checkers already borne off

    Turn generation:
      A turn uses BOTH dice if legally possible.
      If doubles are rolled, the die is used four times.
      If not all dice can be used, the player must use the maximum possible number.
      If exactly one die can be played from a non-double roll, the larger die must be used.


    """

    def __init__(self):
        """
        Create a new environment and initialize the starting position.
        """
        self.reset()

    def reset(self):
        """
        Reset the game to the standard backgammon starting position.

        Returns
        -------
        BackgammonEnv
            The environment itself, which makes reset() convenient to chain.
        """
        self.points = {
            0: [0] * 24,
            1: [0] * 24,
        }

        # Standard backgammon starting position for player 0.
        # Player 0 moves downward: 23 -> 0.
        self.points[0][23] = 2
        self.points[0][12] = 5
        self.points[0][7] = 3
        self.points[0][5] = 5

        # Mirrored starting position for player 1.
        # Player 1 moves upward: 0 -> 23.
        self.points[1][0] = 2
        self.points[1][11] = 5
        self.points[1][16] = 3
        self.points[1][18] = 5

        self.bar = {0: 0, 1: 0}
        self.off = {0: 0, 1: 0}
        self.current_player = 0
        self.turn_count = 0
        return self

    def clone(self) -> "BackgammonEnv":
        """
        Return a deep copy of the environment.

        This is important because afterstate evaluation simulates candidate moves
        without permanently changing the original position.
        """
        other = BackgammonEnv.__new__(BackgammonEnv)
        other.points = {
            0: self.points[0][:],
            1: self.points[1][:],
        }
        other.bar = {0: self.bar[0], 1: self.bar[1]}
        other.off = {0: self.off[0], 1: self.off[1]}
        other.current_player = self.current_player
        other.turn_count = self.turn_count
        return other

    @staticmethod
    def opponent(player: int) -> int:
        """
        Return the opponent index.

        Player 0's opponent is 1.
        Player 1's opponent is 0.
        """
        return 1 - player

    def winner(self) -> Optional[int]:
        """
        Return the winner if the game is over.

        Returns
        -------
        0 or 1 if that player has borne off all 15 checkers.
        None if the game is still ongoing.
        """
        if self.off[0] == 15:
            return 0
        if self.off[1] == 15:
            return 1
        return None

    def roll_dice(self, rng) -> Tuple[int, int]:
        """
        Roll two six-sided dice using the provided random generator.
        """
        return rng.randint(1, 6), rng.randint(1, 6)

    def pip_count(self, player: int) -> int:
        """
        Compute the standard pip count for a player.

        Smaller pip count means the player is closer to bearing off.
        This is used for diagnostics and emergency adjudication when a
        game reaches the hard turn limit.

        Parameters
        ----------
        player:
            Which player's pip count to compute.

        Returns
        -------
        int
            Total pip count.
        """
        total = 0
        if player == 0:
            for pt, count in enumerate(self.points[player]):
                total += count * (pt + 1)
            total += self.bar[player] * 25
        else:
            for pt, count in enumerate(self.points[player]):
                total += count * (24 - pt)
            total += self.bar[player] * 25
        return total

    def in_home_board(self, player: int, point: int) -> bool:
        """
        Check whether a point is inside the player's home board.
        """
        if player == 0:
            return 0 <= point <= 5
        return 18 <= point <= 23

    def all_in_home(self, player: int) -> bool:
        """
        Return True only if all of the player's checkers are in the home board
        and none are on the bar.

        This condition must be satisfied before bearing off is legal.
        """
        if self.bar[player] > 0:
            return False

        for point, count in enumerate(self.points[player]):
            if count == 0:
                continue
            if not self.in_home_board(player, point):
                return False

        return True

    def entry_point_from_bar(self, player: int, die: int) -> int:
        """
        Return the board point where a checker enters from the bar for a given die.
        """
        if player == 0:
            return 24 - die
        return die - 1

    def single_die_legal_moves(self, player: int, die: int) -> List[Tuple[str, int, Optional[int]]]:
        """
        Return all legal checker moves that use exactly ONE die.

        Move encoding:
          (src_kind, src_point, dest_point)

        where:
          - src_kind is "BAR" or "POINT"
          - src_point is -1 for bar, otherwise 0..23
          - dest_point is 0..23, or None for OFF

        Why this function matters:
        A full backgammon turn can contain multiple checker moves.
        To generate full-turn legal play, we repeatedly call this function while
        consuming dice one at a time.
        """
        opp = self.opponent(player)
        legal: List[Tuple[str, int, Optional[int]]] = []

        # If a player has checkers on the bar, those must be entered first.
        if self.bar[player] > 0:
            dest = self.entry_point_from_bar(player, die)
            if self.points[opp][dest] < 2:
                legal.append(("BAR", -1, dest))
            return legal

        # Otherwise the player can move from any occupied point.
        for src in range(24):
            if self.points[player][src] <= 0:
                continue

            if player == 0:
                dest = src - die

                # Normal in-board move.
                if dest >= 0:
                    if self.points[opp][dest] < 2:
                        legal.append(("POINT", src, dest))

                # Bearing off case.
                else:
                    if self.all_in_home(player):
                        # Exact die to bear off.
                        if dest == -1:
                            legal.append(("POINT", src, None))
                        else:
                            # Oversized die can bear off only if no checker is farther away.
                            farther_exists = any(self.points[player][p] > 0 for p in range(src + 1, 6))
                            if not farther_exists:
                                legal.append(("POINT", src, None))

            else:
                dest = src + die

                # Normal in-board move.
                if dest <= 23:
                    if self.points[opp][dest] < 2:
                        legal.append(("POINT", src, dest))

                # Bearing off case.
                else:
                    if self.all_in_home(player):
                        if dest == 24:
                            legal.append(("POINT", src, None))
                        else:
                            farther_exists = any(self.points[player][p] > 0 for p in range(18, src))
                            if not farther_exists:
                                legal.append(("POINT", src, None))

        return legal

    def apply_single_move(self, player: int, move: Tuple[str, int, Optional[int]]):
        """
        Apply exactly one checker move.

        Important:
        This updates the board, bar, hit logic, and off counts,
        but it does NOT switch the turn. A full turn may use multiple dice.
        """
        src_kind, src_point, dest = move
        opp = self.opponent(player)

        if src_kind == "BAR":
            self.bar[player] -= 1
        else:
            self.points[player][src_point] -= 1

        # Bearing off case.
        if dest is None:
            self.off[player] += 1
            return

        # Hitting a blot means exactly one opposing checker is on the destination.
        if self.points[opp][dest] == 1:
            self.points[opp][dest] = 0
            self.bar[opp] += 1

        self.points[player][dest] += 1

    def _generate_turns_for_order(
        self,
        player: int,
        dice_order: Sequence[int],
        idx: int,
        current_state: "BackgammonEnv",
        moves_so_far: List[Tuple[str, int, Optional[int]]],
        used_dice: List[int],
        out: List[Dict],
    ) -> None:
        """
        Recursive helper for generating all full-turn outcomes for a fixed dice order.

        Example:
          [6, 1] and [1, 6] can lead to different legal positions.
          Doubles become something like [4, 4, 4, 4].

        This function keeps partial results too, because a player may be able to
        use some dice but not all of them. The official backgammon rules are
        enforced later when we keep only the best legal candidates.
        """
        if idx >= len(dice_order):
            out.append(
                {
                    "state": current_state,
                    "moves": moves_so_far[:],
                    "used_dice": used_dice[:],
                }
            )
            return

        die = dice_order[idx]
        legal = current_state.single_die_legal_moves(player, die)

        # If this die cannot be used at this stage, the turn ends here.
        if not legal:
            out.append(
                {
                    "state": current_state,
                    "moves": moves_so_far[:],
                    "used_dice": used_dice[:],
                }
            )
            return

        for move in legal:
            nxt = current_state.clone()
            nxt.apply_single_move(player, move)
            self._generate_turns_for_order(
                player=player,
                dice_order=dice_order,
                idx=idx + 1,
                current_state=nxt,
                moves_so_far=moves_so_far + [move],
                used_dice=used_dice + [die],
                out=out,
            )

    def legal_turn_afterstates(self, player: int, dice: Tuple[int, int]) -> List[Dict]:
        """
        Generate legal FULL-TURN afterstates for a dice roll.

        This function handles the tricky real backgammon rules:
          1. dice order matters
          2. doubles mean four moves
          3. maximum number of dice must be used if possible
          4. if only one die can be used from a non-double, the larger die must be used

        Returns
        -------
        List[Dict]
            Each element contains:
              - "state": the resulting afterstate, with turn already passed
              - "moves": the checker moves used
              - "used_dice": which dice were consumed

        Why "afterstate" matters:
        TD-Gammon commonly evaluates positions after the player's move is complete,
        not action-values for every tiny step.
        """
        d1, d2 = dice

        if d1 == d2:
            orders = [[d1, d1, d1, d1]]
        else:
            orders = [[d1, d2], [d2, d1]]

        candidates: List[Dict] = []

        for order in orders:
            raw: List[Dict] = []
            self._generate_turns_for_order(
                player=player,
                dice_order=order,
                idx=0,
                current_state=self.clone(),
                moves_so_far=[],
                used_dice=[],
                out=raw,
            )
            candidates.extend(raw)

        # Fallback pure pass turn.
        if not candidates:
            passed = self.clone()
            passed.current_player = self.opponent(player)
            passed.turn_count += 1
            return [{"state": passed, "moves": [], "used_dice": []}]

        # Keep only candidates that use the maximum number of dice.
        max_used = max(len(c["used_dice"]) for c in candidates)
        candidates = [c for c in candidates if len(c["used_dice"]) == max_used]

        # Official tiebreak for non-doubles:
        # if only one die can be played, the larger die must be used.
        if d1 != d2 and max_used == 1:
            larger = max(d1, d2)
            larger_candidates = [c for c in candidates if c["used_dice"] and c["used_dice"][0] == larger]
            if larger_candidates:
                candidates = larger_candidates

        # Deduplicate resulting afterstates.
        seen = set()
        deduped = []

        for c in candidates:
            c_state = c["state"].clone()

            # Afterstate means the move is complete and it is now the opponent's turn.
            c_state.current_player = self.opponent(player)
            c_state.turn_count += 1

            key = (
                tuple(c_state.points[0]),
                tuple(c_state.points[1]),
                c_state.bar[0],
                c_state.bar[1],
                c_state.off[0],
                c_state.off[1],
                c_state.current_player,
            )

            if key in seen:
                continue

            seen.add(key)
            deduped.append({"state": c_state, "moves": c["moves"], "used_dice": c["used_dice"]})

        return deduped

    def outcome_reward_for_player(self, player: int) -> float:
        """
        Return terminal reward from one player's perspective.

        Rewards:
        - normal win: +1
        - gammon: +2
        - backgammon: +3
        - normal loss: -1
        - gammon loss: -2
        - backgammon loss: -3

        This makes the terminal signal more similar to stronger TD-Gammon-style
        experimentation than plain win/loss only.
        """
        winner = self.winner()
        if winner is None:
            raise ValueError("outcome_reward_for_player() called on nonterminal state")

        loser = self.opponent(winner)

        # Determine whether the loser bore off any checkers.
        loser_off = self.off[loser]

        # Check whether loser still has a checker on the bar or in the winner's home board.
        loser_on_bar = self.bar[loser] > 0

        if loser == 0:
            loser_in_winner_home = any(self.points[loser][p] > 0 for p in range(18, 24))
        else:
            loser_in_winner_home = any(self.points[loser][p] > 0 for p in range(0, 6))

        if loser_off > 0:
            magnitude = 1.0
        else:
            if loser_on_bar or loser_in_winner_home:
                magnitude = 3.0
            else:
                magnitude = 2.0

        return magnitude if player == winner else -magnitude

    def encode_for_player(self, player: int) -> np.ndarray:
        """
        Encode the board from ONE player's perspective.

        This is the most TD-Gammon-like part of the representation.

        We do NOT use a tabular value table because backgammon has a huge state space.
        Instead we convert the board into a fixed-size feature vector and let the
        neural network approximate the value function.

        Feature design:
        - 4 features for own checkers on each point
        - 4 features for opponent checkers on each point
        - 24 points total
        - 8 * 24 = 192 point features
        - plus 6 global features
        - total = 198 features

        Unary-style point encoding:
          f1 = 1 if at least 1 checker is on the point
          f2 = 1 if at least 2 checkers
          f3 = 1 if at least 3 checkers
          f4 = scaled extra amount above 3 checkers

        The board is re-oriented so the current player always sees the game from
        their own direction of movement. This lets one shared network work for
        both sides during self-play.
        """
        opp = self.opponent(player)
        feats: List[float] = []

        def point_features(n: int) -> List[float]:
            """
            Convert a checker count on one point into 4 features.
            """
            return [
                1.0 if n >= 1 else 0.0,
                1.0 if n >= 2 else 0.0,
                1.0 if n >= 3 else 0.0,
                max(0.0, (n - 3) / 2.0),
            ]

        # Re-orient the board so the current player always sees movement
        # in a consistent direction in feature space.
        if player == 0:
            own_view = list(range(23, -1, -1))
            opp_view = list(range(0, 24))
        else:
            own_view = list(range(0, 24))
            opp_view = list(range(23, -1, -1))

        for idx, pt in enumerate(own_view):
            feats.extend(point_features(self.points[player][pt]))
            feats.extend(point_features(self.points[opp][opp_view[idx]]))

        feats.extend(
            [
                self.bar[player] / 2.0,
                self.bar[opp] / 2.0,
                self.off[player] / 15.0,
                self.off[opp] / 15.0,
                self.pip_count(player) / 200.0,
                self.pip_count(opp) / 200.0,
            ]
        )

        return np.array(feats, dtype=np.float64)

    def pretty_move(self, player: int, move: Tuple[str, int, Optional[int]]) -> str:
        """
        Convert one move into readable text for logs and demo output.
        """
        src_kind, src_point, dest = move

        if src_kind == "BAR":
            src_text = "BAR"
        else:
            src_text = str(src_point)

        dest_text = "OFF" if dest is None else str(dest)
        side = "P0" if player == 0 else "P1"
        return f"{side}: {src_text} -> {dest_text}"