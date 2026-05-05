# Skull ML Training Project

Multi-agent reinforcement learning system to train AI agents to play the bluffing game Skull.

## Project Structure

```
skull/
  env/
    skull_env.py        # PettingZoo AECEnv wrapper
    game_state.py       # Pure game logic (no ML, no side effects)
    obs_encoder.py      # Game state -> observation vector
  training/
    train_selfplay.py   # RLlib + PPO + self-play config
  tests/
    test_random.py      # Random agent sanity checks
  skull_ruleset.txt     # Full ML-friendly ruleset (source of truth)
```

## Commands

```bash
pip install pettingzoo ray[rllib] torch
python tests/test_random.py        # sanity check env
python training/train_selfplay.py  # start training
```

## Game Summary

- 2–6 players, each with 4 discs: 3 flowers + 1 skull
- Each round: players place discs face-down, then bid on how many they can flip without hitting a skull
- Highest bidder (Challenger) must flip that many discs — their own stack first, then opponents' in any order
- Flip a skull = lose a disc permanently. Succeed = flip mat. Two successes = win.
- Core challenge for ML: partial observability + bluffing (opponent disc contents are hidden)

Full rules: see `skull_ruleset.txt`

## Environment Spec

### Observation Space (per agent)
Each agent receives a vector of public state + their own private state:
- Public: disc counts per player, stack sizes, mat orientations (wins), current bid, who has passed
- Private: contents of own hand, contents of own stack (ordered)
- Hidden (not in obs): contents of opponent stacks

### Action Space (phase-dependent)
- Phase 1 (Placement):   PLACE(disc)      — disc ∈ {flower=0, skull=1}
- Phase 2 (Add or Bid):  PLACE(disc) or BID(n)
- Phase 3 (Bidding):     RAISE(n) or PASS
- Phase 4 (Attempt):     FLIP(player_id)  — flip top of that player's stack

Use action masking to restrict illegal actions per phase.

### Rewards
```python
WIN_GAME         = +1.0
SUCCEED_CHALLENGE = +0.3
FAIL_CHALLENGE   = -0.3
LOSE_DISC        = -0.1
ELIMINATE_OPPONENT = +0.2
BE_ELIMINATED    = -1.0
```

## Architecture Conventions

- `game_state.py` must be pure logic — no RL, no randomness beyond disc shuffle, fully deterministic given seed
- `skull_env.py` wraps game_state as a PettingZoo `AECEnv` (turn-based, not parallel)
- Never leak hidden information into the observation vector — opponent stack contents are always masked
- All disc identities are integers: flower=0, skull=1
- Player IDs are 0-indexed integers
- Stacks are lists ordered bottom-to-top; index[-1] = top disc (first to be flipped)

## Framework Stack

| Layer | Library | Notes |
|---|---|---|
| Environment | PettingZoo AECEnv | Standard for turn-based multi-agent |
| Training | RLlib (Ray) | Self-play support built-in |
| Algorithm | PPO | Start here; switch to CFR via OpenSpiel for stronger bluffing |
| Models | PyTorch | Default backend |

## Training Approach

1. **Phase 1 — Random baseline**: Run 10k games with random agents. Verify uniform win rates and no illegal states.
2. **Phase 2 — Self-play PPO**: Train with RLlib self-play. Agents play copies of themselves.
3. **Phase 3 — Belief state (optional)**: Add a belief module that tracks probability distributions over hidden opponent discs.
4. **Phase 4 — CFR (optional)**: Port to OpenSpiel for Counterfactual Regret Minimization — theoretically optimal for bluffing games.

## Known Hard Problems

- **Bluffing**: A pure reward-maximizing RL agent may converge to exploitable deterministic strategies. CFR handles this better.
- **Partial observability**: Agents must infer opponent disc contents from betting behavior. Consider LSTM or attention over action history.
- **Disc loss asymmetry**: Losing the skull disc vs. a flower disc has very different strategic implications — the agent can't always know which it lost.

## Code Style

- Python 3.10+
- Type hints everywhere
- Keep game_state.py free of any library imports beyond stdlib
- Tests use pytest
- No notebooks — scripts only
