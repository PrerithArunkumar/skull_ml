"""PettingZoo AECEnv wrapper around GameState.

Per-agent observation is a Dict:
    {
      "observation": Box(shape=(OBS_SIZE,)) — float vector,
      "action_mask": Box(shape=(ACTION_SPACE_SIZE,), int8) — 1 = legal,
    }

Action space is a fixed Discrete(ACTION_SPACE_SIZE). Illegal actions raise; train
agents must respect "action_mask" (or use a masking policy/model).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from .game_state import GameState, Phase
from .obs_encoder import (
    ACTION_SPACE_SIZE,
    OBS_SIZE,
    action_mask,
    decode_action,
    encode_observation,
)


def _agent_name(pid: int) -> str:
    return f"player_{pid}"


def _pid_from_agent(agent: str) -> int:
    return int(agent.split("_", 1)[1])


class SkullEnv(AECEnv):
    metadata = {"render_modes": [], "name": "skull_v0", "is_parallelizable": False}

    def __init__(self, num_players: int = 4, seed: Optional[int] = None):
        super().__init__()
        self._num_players = num_players
        self._seed = seed
        self.possible_agents: list[str] = [_agent_name(i) for i in range(num_players)]
        self.agents: list[str] = list(self.possible_agents)

        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8
                ),
            }
        )
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {
            a: spaces.Discrete(ACTION_SPACE_SIZE) for a in self.possible_agents
        }

        self.game: Optional[GameState] = None
        self._agent_selector: Optional[agent_selector] = None

    # ---- Spaces ------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ---- Reset / step ------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        if seed is not None:
            self._seed = seed
        self.game = GameState(self._num_players, seed=self._seed)
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos: dict[str, dict[str, Any]] = {a: {} for a in self.agents}
        self.agent_selection = _agent_name(self.game.current_player)

    def step(self, action: int) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        assert self.game is not None
        agent = self.agent_selection
        pid = _pid_from_agent(agent)
        if pid != self.game.current_player:
            raise RuntimeError(
                f"agent_selection {agent} (pid {pid}) != game.current_player "
                f"{self.game.current_player}"
            )

        decoded = decode_action(int(action), self.game.phase)
        deltas = self.game.step(decoded)

        # Reset per-step rewards then assign deltas.
        self.rewards = {a: 0.0 for a in self.agents}
        for p, r in deltas.items():
            self.rewards[_agent_name(p)] = float(r)

        # Mark eliminated / game-over agents as terminated.
        for p in range(self._num_players):
            if not self.game.alive[p]:
                self.terminations[_agent_name(p)] = True
        if self.game.phase == Phase.GAME_OVER:
            for a in self.agents:
                self.terminations[a] = True

        # Advance agent_selection to whoever is next on turn (only if game continues).
        if self.game.phase != Phase.GAME_OVER:
            self.agent_selection = _agent_name(self.game.current_player)

        self._cumulative_rewards[agent] = 0.0
        self._accumulate_rewards()

    # ---- Observation -------------------------------------------------------

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        assert self.game is not None
        pid = _pid_from_agent(agent)
        obs = np.asarray(encode_observation(self.game, pid), dtype=np.float32)
        mask = np.asarray(action_mask(self.game), dtype=np.int8)
        return {"observation": obs, "action_mask": mask}

    # ---- Misc --------------------------------------------------------------

    def render(self) -> None:
        if self.game is None:
            print("(env not reset)")
            return
        g = self.game
        print(
            f"phase={g.phase.name} cur={g.current_player} bid={g.current_bid} "
            f"bidder={g.bidder} challenger={g.challenger} winner={g.winner}"
        )
        for p in range(g.num_players):
            print(
                f"  p{p} hand={len(g.hands[p])} stack={g.stacks[p]} "
                f"flipped={g.flipped[p]} wins={g.wins[p]} alive={g.alive[p]} "
                f"passed={g.passed[p]}"
            )

    def close(self) -> None:
        self.game = None
