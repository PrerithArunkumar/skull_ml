"""Encode GameState into per-agent observation vector + action mask.

Hidden information (opponent hand contents, opponent face-down stack contents) is
never written into the observation. Face-up flipped discs ARE public and exposed
as counts.

Action ID layout (size = ACTION_SPACE_SIZE = 33):
  0           PLACE_FLOWER
  1           PLACE_SKULL
  2           PASS
  3..26       BID/RAISE n, where n = action_id - 2  (n in 1..24)
  27..32      FLIP target = action_id - 27           (target in 0..5)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .game_state import GameState

from .game_state import (
    Action,
    ActionType,
    FLOWER,
    MAX_PLAYERS,
    Phase,
    SKULL,
)


ACTION_PLACE_FLOWER: int = 0
ACTION_PLACE_SKULL: int = 1
ACTION_PASS: int = 2
ACTION_BID_BASE: int = 3       # action_id 3..26 → n = id - 2  (1..24)
ACTION_FLIP_BASE: int = 27     # action_id 27..32 → target = id - 27  (0..5)
MAX_BID: int = MAX_PLAYERS * 4  # 24
ACTION_SPACE_SIZE: int = ACTION_FLIP_BASE + MAX_PLAYERS  # 33

# Observation vector layout — fixed regardless of num_players.
PER_PLAYER_PUBLIC_FEATURES: int = 7
GLOBAL_FEATURES: int = 4 + 1 + MAX_PLAYERS * 4  # phase + bid + 4 player-id one-hots
PRIVATE_FEATURES: int = 2 + 4 * 2               # hand counts + own stack one-hot
OBS_SIZE: int = (
    PER_PLAYER_PUBLIC_FEATURES * MAX_PLAYERS + GLOBAL_FEATURES + PRIVATE_FEATURES
)


def encode_action(action: Action) -> int:
    if action.type == ActionType.PLACE_FLOWER:
        return ACTION_PLACE_FLOWER
    if action.type == ActionType.PLACE_SKULL:
        return ACTION_PLACE_SKULL
    if action.type == ActionType.PASS:
        return ACTION_PASS
    if action.type in (ActionType.BID, ActionType.RAISE):
        return ACTION_BID_BASE + (action.value - 1)
    if action.type == ActionType.FLIP:
        return ACTION_FLIP_BASE + action.value
    raise ValueError(f"cannot encode action {action}")


def decode_action(action_id: int, phase: Phase) -> Action:
    if action_id == ACTION_PLACE_FLOWER:
        return Action(ActionType.PLACE_FLOWER)
    if action_id == ACTION_PLACE_SKULL:
        return Action(ActionType.PLACE_SKULL)
    if action_id == ACTION_PASS:
        return Action(ActionType.PASS)
    if ACTION_BID_BASE <= action_id < ACTION_FLIP_BASE:
        n = action_id - ACTION_BID_BASE + 1
        if phase == Phase.BIDDING:
            return Action(ActionType.RAISE, n)
        return Action(ActionType.BID, n)
    if ACTION_FLIP_BASE <= action_id < ACTION_SPACE_SIZE:
        return Action(ActionType.FLIP, action_id - ACTION_FLIP_BASE)
    raise ValueError(f"out-of-range action_id {action_id}")


def action_mask(state: "GameState") -> list[int]:
    mask = [0] * ACTION_SPACE_SIZE
    for a in state.legal_actions():
        mask[encode_action(a)] = 1
    return mask


def encode_observation(state: "GameState", agent_id: int) -> list[float]:
    """Return a flat float vector of length OBS_SIZE from agent_id's perspective."""
    obs: list[float] = []

    # ---- Per-player public features (MAX_PLAYERS slots; unused slots are zero).
    for pid in range(MAX_PLAYERS):
        if pid >= state.num_players:
            obs.extend([0.0] * PER_PLAYER_PUBLIC_FEATURES)
            continue
        flipped = state.flipped[pid]
        flower_revealed = sum(1 for d in flipped if d == FLOWER)
        skull_revealed = 1.0 if any(d == SKULL for d in flipped) else 0.0
        obs.extend(
            [
                len(state.hands[pid]) / 4.0,
                len(state.stacks[pid]) / 4.0,
                state.wins[pid] / 1.0,
                1.0 if state.alive[pid] else 0.0,
                1.0 if state.passed[pid] else 0.0,
                flower_revealed / 4.0,
                skull_revealed,
            ]
        )

    # ---- Global features.
    phase_oh = [0.0] * 4
    if state.phase in (Phase.PLACEMENT, Phase.ADD_OR_BID, Phase.BIDDING, Phase.ATTEMPT):
        phase_oh[int(state.phase) - 1] = 1.0
    obs.extend(phase_oh)
    obs.append(state.current_bid / float(MAX_BID))
    for value in (state.bidder, state.challenger, state.first_player, state.current_player):
        oh = [0.0] * MAX_PLAYERS
        if 0 <= value < MAX_PLAYERS:
            oh[value] = 1.0
        obs.extend(oh)

    # ---- Private features (own hand + own stack contents).
    own_hand = state.hands[agent_id]
    obs.append(sum(1 for d in own_hand if d == FLOWER) / 3.0)
    obs.append(1.0 if SKULL in own_hand else 0.0)

    own_stack = state.stacks[agent_id]
    for slot in range(4):
        if slot < len(own_stack):
            disc = own_stack[slot]
            obs.append(1.0 if disc == FLOWER else 0.0)
            obs.append(1.0 if disc == SKULL else 0.0)
        else:
            obs.extend([0.0, 0.0])

    assert len(obs) == OBS_SIZE, f"obs size {len(obs)} != {OBS_SIZE}"
    return obs
