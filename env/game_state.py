"""Pure Skull game logic. Stdlib-only. Deterministic given a seed.

Disc identities: flower=0, skull=1.
Player IDs: 0-indexed integers.
Stacks: bottom-to-top, index[-1] = top disc (first to be flipped).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


FLOWER: int = 0
SKULL: int = 1

WIN_GAME: float = 1.0
SUCCEED_CHALLENGE: float = 0.3
FAIL_CHALLENGE: float = -0.3
LOSE_DISC: float = -0.1
ELIMINATE_OPPONENT: float = 0.2
BE_ELIMINATED: float = -1.0

MAX_PLAYERS: int = 6
DISCS_PER_PLAYER: int = 4


class Phase(IntEnum):
    PLACEMENT = 1
    ADD_OR_BID = 2
    BIDDING = 3
    ATTEMPT = 4
    GAME_OVER = 5


class ActionType(IntEnum):
    PLACE_FLOWER = 0
    PLACE_SKULL = 1
    PASS = 2
    BID = 3
    RAISE = 4
    FLIP = 5


@dataclass(frozen=True)
class Action:
    type: ActionType
    value: int = 0  # bid amount for BID/RAISE, target player_id for FLIP


class GameState:
    def __init__(self, num_players: int, seed: Optional[int] = None):
        if not (2 <= num_players <= MAX_PLAYERS):
            raise ValueError(f"num_players must be in [2,{MAX_PLAYERS}], got {num_players}")
        self.num_players: int = num_players
        self.rng: random.Random = random.Random(seed)
        self.hands: list[list[int]] = [
            [FLOWER, FLOWER, FLOWER, SKULL] for _ in range(num_players)
        ]
        self.stacks: list[list[int]] = [[] for _ in range(num_players)]
        self.flipped: list[list[int]] = [[] for _ in range(num_players)]
        self.wins: list[int] = [0] * num_players
        self.alive: list[bool] = [True] * num_players
        self.passed: list[bool] = [False] * num_players
        self.placed_this_round: list[bool] = [False] * num_players
        self.phase: Phase = Phase.PLACEMENT
        self.first_player: int = self.rng.randrange(num_players)
        self.current_player: int = self.first_player
        self.current_bid: int = 0
        self.bidder: int = -1
        self.challenger: int = -1
        self.flips_this_attempt: int = 0
        self.winner: Optional[int] = None

    # ---- Public queries ----------------------------------------------------

    def is_terminal(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def total_stack_discs(self) -> int:
        return sum(len(s) for s in self.stacks)

    def legal_actions(self) -> list[Action]:
        if self.phase == Phase.GAME_OVER:
            return []
        p = self.current_player
        if self.phase == Phase.PLACEMENT:
            return self._place_actions(p)
        if self.phase == Phase.ADD_OR_BID:
            actions: list[Action] = []
            if self.hands[p]:
                actions.extend(self._place_actions(p))
            max_bid = self.total_stack_discs()
            for n in range(1, max_bid + 1):
                actions.append(Action(ActionType.BID, n))
            return actions
        if self.phase == Phase.BIDDING:
            actions = [Action(ActionType.PASS)]
            max_bid = self.total_stack_discs()
            for n in range(self.current_bid + 1, max_bid + 1):
                actions.append(Action(ActionType.RAISE, n))
            return actions
        if self.phase == Phase.ATTEMPT:
            if self.stacks[p]:
                return [Action(ActionType.FLIP, p)]
            return [
                Action(ActionType.FLIP, i)
                for i in range(self.num_players)
                if i != p and self.alive[i] and self.stacks[i]
            ]
        return []

    def _place_actions(self, p: int) -> list[Action]:
        out: list[Action] = []
        if FLOWER in self.hands[p]:
            out.append(Action(ActionType.PLACE_FLOWER))
        if SKULL in self.hands[p]:
            out.append(Action(ActionType.PLACE_SKULL))
        return out

    # ---- Step --------------------------------------------------------------

    def step(self, action: Action) -> dict[int, float]:
        """Apply action by current_player. Returns per-player reward deltas."""
        if self.phase == Phase.GAME_OVER:
            raise RuntimeError("step called after game over")
        rewards: dict[int, float] = {i: 0.0 for i in range(self.num_players)}
        if self.phase == Phase.PLACEMENT:
            self._step_placement(action)
        elif self.phase == Phase.ADD_OR_BID:
            self._step_add_or_bid(action)
        elif self.phase == Phase.BIDDING:
            self._step_bidding(action)
        elif self.phase == Phase.ATTEMPT:
            self._step_attempt(action, rewards)
        return rewards

    def _step_placement(self, action: Action) -> None:
        p = self.current_player
        disc = self._place_disc_from_action(action, p)
        self.hands[p].remove(disc)
        self.stacks[p].append(disc)
        self.placed_this_round[p] = True
        if all(
            self.placed_this_round[i] or not self.alive[i]
            for i in range(self.num_players)
        ):
            self.phase = Phase.ADD_OR_BID
            self.current_player = self.first_player
        else:
            self.current_player = self._next_alive(p)

    def _step_add_or_bid(self, action: Action) -> None:
        p = self.current_player
        if action.type in (ActionType.PLACE_FLOWER, ActionType.PLACE_SKULL):
            if not self.hands[p]:
                raise ValueError("cannot PLACE with empty hand")
            disc = self._place_disc_from_action(action, p)
            self.hands[p].remove(disc)
            self.stacks[p].append(disc)
            self.current_player = self._next_alive(p)
            return
        if action.type == ActionType.BID:
            if not (1 <= action.value <= self.total_stack_discs()):
                raise ValueError(f"illegal bid {action.value}")
            self.current_bid = action.value
            self.bidder = p
            self.phase = Phase.BIDDING
            self.current_player = self._next_active_bidder(p)
            return
        raise ValueError(f"illegal action {action} in ADD_OR_BID")

    def _step_bidding(self, action: Action) -> None:
        p = self.current_player
        if action.type == ActionType.RAISE:
            if action.value <= self.current_bid or action.value > self.total_stack_discs():
                raise ValueError(f"illegal raise {action.value} (bid={self.current_bid})")
            self.current_bid = action.value
            self.bidder = p
            nxt = self._next_active_bidder(p)
            if nxt == p:
                self._start_attempt(p)
            else:
                self.current_player = nxt
            return
        if action.type == ActionType.PASS:
            self.passed[p] = True
            active = [
                i for i in range(self.num_players)
                if self.alive[i] and not self.passed[i]
            ]
            if len(active) == 1:
                self._start_attempt(active[0])
            else:
                self.current_player = self._next_active_bidder(p)
            return
        raise ValueError(f"illegal action {action} in BIDDING")

    def _step_attempt(self, action: Action, rewards: dict[int, float]) -> None:
        if action.type != ActionType.FLIP:
            raise ValueError(f"illegal action {action} in ATTEMPT")
        target = action.value
        challenger = self.challenger
        if self.stacks[challenger] and target != challenger:
            raise ValueError("must clear own stack before flipping opponents")
        if not (0 <= target < self.num_players):
            raise ValueError(f"illegal target {target}")
        if not self.alive[target] or not self.stacks[target]:
            raise ValueError(f"target {target} has no flippable stack")

        disc = self.stacks[target].pop()
        self.flipped[target].append(disc)

        if disc == SKULL:
            self._resolve_failure(target, rewards)
        else:
            self.flips_this_attempt += 1
            if self.flips_this_attempt >= self.current_bid:
                self._resolve_success(rewards)

    # ---- Resolution helpers ------------------------------------------------

    def _start_attempt(self, challenger: int) -> None:
        self.phase = Phase.ATTEMPT
        self.challenger = challenger
        self.current_player = challenger
        self.flips_this_attempt = 0

    def _resolve_failure(self, skull_owner: int, rewards: dict[int, float]) -> None:
        challenger = self.challenger
        rewards[challenger] += FAIL_CHALLENGE

        # Special case: a non-challenger whose only remaining disc was the just-flipped
        # skull is eliminated (rules: "1 DISC REMAINING" clause).
        special_elim = False
        if skull_owner != challenger:
            total_owner = (
                len(self.hands[skull_owner])
                + len(self.stacks[skull_owner])
                + len(self.flipped[skull_owner])
            )
            if total_owner == 1:
                special_elim = True

        # Return all face-down and face-up discs to hand, except an eliminated owner's.
        for i in range(self.num_players):
            if not self.alive[i]:
                continue
            if i == skull_owner and special_elim:
                self.flipped[i].clear()
                self.stacks[i].clear()
                continue
            self.hands[i].extend(self.stacks[i])
            self.hands[i].extend(self.flipped[i])
            self.stacks[i].clear()
            self.flipped[i].clear()

        if special_elim:
            self._eliminate(skull_owner, rewards)
            rewards[challenger] += ELIMINATE_OPPONENT

        # Challenger penalty: lose one disc.
        if self.alive[challenger] and self.hands[challenger]:
            self._challenger_loses_disc(challenger, skull_owner, rewards)

        if self._check_game_over(rewards):
            return

        self._start_next_round(
            self._next_first_player_after_failure(challenger, skull_owner, special_elim)
        )

    def _challenger_loses_disc(
        self, challenger: int, skull_owner: int, rewards: dict[int, float]
    ) -> None:
        discs = list(self.hands[challenger])
        if skull_owner == challenger:
            # Challenger picks; lose a flower if available (skull is the bluff piece).
            removed = FLOWER if FLOWER in discs else SKULL
        else:
            removed = self.rng.choice(discs)
        self.hands[challenger].remove(removed)
        rewards[challenger] += LOSE_DISC
        if not self.hands[challenger]:
            self._eliminate(challenger, rewards)
            if skull_owner != challenger:
                rewards[skull_owner] += ELIMINATE_OPPONENT

    def _resolve_success(self, rewards: dict[int, float]) -> None:
        challenger = self.challenger
        rewards[challenger] += SUCCEED_CHALLENGE
        if self.wins[challenger] >= 1:
            rewards[challenger] += WIN_GAME
            self.winner = challenger
            self.phase = Phase.GAME_OVER
            return
        self.wins[challenger] = 1
        for i in range(self.num_players):
            if not self.alive[i]:
                continue
            self.hands[i].extend(self.stacks[i])
            self.hands[i].extend(self.flipped[i])
            self.stacks[i].clear()
            self.flipped[i].clear()
        self._start_next_round(challenger)

    def _eliminate(self, p: int, rewards: dict[int, float]) -> None:
        self.alive[p] = False
        self.hands[p].clear()
        self.stacks[p].clear()
        self.flipped[p].clear()
        self.wins[p] = 0
        rewards[p] += BE_ELIMINATED

    def _check_game_over(self, rewards: dict[int, float]) -> bool:
        alive_ids = [i for i in range(self.num_players) if self.alive[i]]
        if len(alive_ids) <= 1:
            if alive_ids:
                self.winner = alive_ids[0]
                rewards[alive_ids[0]] += WIN_GAME
            self.phase = Phase.GAME_OVER
            return True
        return False

    def _start_next_round(self, first_player: int) -> None:
        if not self.alive[first_player]:
            first_player = self._next_alive(first_player)
        self.phase = Phase.PLACEMENT
        self.first_player = first_player
        self.current_player = first_player
        self.current_bid = 0
        self.bidder = -1
        self.challenger = -1
        self.flips_this_attempt = 0
        self.passed = [False] * self.num_players
        self.placed_this_round = [False] * self.num_players

    def _next_first_player_after_failure(
        self, challenger: int, skull_owner: int, special_elim: bool
    ) -> int:
        if self.alive[challenger]:
            return challenger
        # Challenger eliminated.
        if skull_owner == challenger:
            # "Challenger's choice" — pick next alive clockwise as a deterministic stand-in.
            return self._next_alive(challenger)
        if self.alive[skull_owner]:
            return skull_owner
        return self._next_alive(challenger)

    # ---- Rotation helpers --------------------------------------------------

    def _next_alive(self, p: int) -> int:
        n = self.num_players
        for i in range(1, n + 1):
            cand = (p + i) % n
            if self.alive[cand]:
                return cand
        return p

    def _next_active_bidder(self, p: int) -> int:
        n = self.num_players
        for i in range(1, n + 1):
            cand = (p + i) % n
            if self.alive[cand] and not self.passed[cand]:
                return cand
        return p

    # ---- Internal ----------------------------------------------------------

    @staticmethod
    def _place_disc_from_action(action: Action, p: int) -> int:
        if action.type == ActionType.PLACE_FLOWER:
            return FLOWER
        if action.type == ActionType.PLACE_SKULL:
            return SKULL
        raise ValueError(f"not a placement action: {action}")
