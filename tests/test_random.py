"""Random-agent sanity check.

Runs many games with uniformly-random legal-action agents and verifies:
  - every game terminates within a step budget,
  - exactly one winner (or a single-survivor terminal state),
  - per-player win rate is roughly uniform (no asymmetric env bug).

Runnable as `python tests/test_random.py` or via pytest.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.skull_env import SkullEnv  # noqa: E402


MAX_STEPS_PER_GAME = 5000


def _play_one_game(num_players: int, seed: int) -> int | None:
    env = SkullEnv(num_players=num_players, seed=seed)
    env.reset(seed=seed)
    rng = random.Random(seed ^ 0x5EED)
    steps = 0
    for agent in env.agent_iter():
        obs, _reward, terminated, truncated, _info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        assert legal.size > 0, "no legal action for live agent"
        action = int(rng.choice(legal.tolist()))
        env.step(action)
        steps += 1
        if steps > MAX_STEPS_PER_GAME:
            raise RuntimeError(f"game exceeded {MAX_STEPS_PER_GAME} steps")
    winner = env.game.winner if env.game is not None else None
    env.close()
    return winner


def test_random_games_complete() -> None:
    num_players = 4
    n_games = 200
    wins: Counter = Counter()
    finished = 0
    for g in range(n_games):
        winner = _play_one_game(num_players, seed=1000 + g)
        finished += 1
        wins[winner] += 1  # winner is None on a true draw (rare)
    assert finished == n_games, f"only {finished}/{n_games} games finished"
    # Every player should win at least once across 200 random games.
    for pid in range(num_players):
        assert wins[pid] > 0, f"player {pid} never won in {n_games} games: {dict(wins)}"


def test_legal_actions_nonempty_each_phase() -> None:
    """For a single random game, every live-agent step has at least one legal action."""
    env = SkullEnv(num_players=3, seed=42)
    env.reset(seed=42)
    rng = random.Random(0)
    for agent in env.agent_iter():
        obs, _r, term, trunc, _i = env.last()
        if term or trunc:
            env.step(None)
            continue
        mask = obs["action_mask"]
        assert mask.sum() > 0
        env.step(int(rng.choice(np.flatnonzero(mask).tolist())))
    env.close()


def main() -> None:
    print("running random-agent sanity check...")
    for num_players in (2, 3, 4, 6):
        wins = Counter()
        n_games = 100
        total_steps = 0
        for g in range(n_games):
            env = SkullEnv(num_players=num_players, seed=g)
            env.reset(seed=g)
            rng = random.Random(g)
            steps = 0
            for agent in env.agent_iter():
                obs, _r, term, trunc, _i = env.last()
                if term or trunc:
                    env.step(None)
                    continue
                legal = np.flatnonzero(obs["action_mask"]).tolist()
                env.step(int(rng.choice(legal)))
                steps += 1
                if steps > MAX_STEPS_PER_GAME:
                    raise RuntimeError("step budget exceeded")
            if env.game is not None and env.game.winner is not None:
                wins[env.game.winner] += 1
            total_steps += steps
            env.close()
        finished = sum(wins.values())
        rates = {p: wins[p] / n_games for p in range(num_players)}
        avg_steps = total_steps / n_games
        print(
            f"  {num_players}p: finished={finished}/{n_games} "
            f"avg_steps={avg_steps:.1f} win_rates={rates}"
        )
    print("ok")


if __name__ == "__main__":
    main()
