"""RLlib PPO self-play config for SkullEnv.

A single shared policy is mapped to every agent — the canonical "self-play" setup
for symmetric multi-agent games. The custom torch model masks illegal actions by
adding -inf to their logits before sampling.

Run:
    python training/train_selfplay.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.obs_encoder import ACTION_SPACE_SIZE, OBS_SIZE  # noqa: E402
from env.skull_env import SkullEnv  # noqa: E402


ENV_NAME = "skull"
POLICY_ID = "shared"


def _env_creator(env_config):
    return PettingZooEnv(
        SkullEnv(
            num_players=env_config.get("num_players", 4),
            seed=env_config.get("seed", None),
        )
    )


class ActionMaskedModel(TorchModelV2, nn.Module):
    """MLP over `observation` that masks illegal actions via `action_mask`."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        hidden = model_config.get("custom_model_config", {}).get("hidden", 256)
        self.trunk = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, ACTION_SPACE_SIZE)
        self.value_head = nn.Linear(hidden, 1)
        self._value_out: torch.Tensor | None = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"].float()
        mask = input_dict["obs"]["action_mask"].float()
        h = self.trunk(obs)
        logits = self.policy_head(h)
        # Add a large negative bias to illegal actions; legal stay unchanged.
        logits = logits + torch.clamp(torch.log(mask), min=FLOAT_MIN)
        self._value_out = self.value_head(h).squeeze(-1)
        return logits, state

    def value_function(self):
        assert self._value_out is not None, "must call forward() before value_function()"
        return self._value_out


def build_config(num_players: int = 4) -> PPOConfig:
    return (
        PPOConfig()
        .environment(env=ENV_NAME, env_config={"num_players": num_players})
        .framework("torch")
        .multi_agent(
            policies={POLICY_ID},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: POLICY_ID,
        )
        .training(
            model={"custom_model": "action_masked", "custom_model_config": {"hidden": 256}},
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            gamma=0.99,
            lr=3e-4,
            entropy_coeff=0.01,
        )
        .env_runners(num_env_runners=2, rollout_fragment_length="auto")
        .resources(num_gpus=int(os.environ.get("NUM_GPUS", "0")))
    )


def main(num_iters: int = 100, num_players: int = 4) -> None:
    ray.init(ignore_reinit_error=True)
    tune.register_env(ENV_NAME, _env_creator)
    ModelCatalog.register_custom_model("action_masked", ActionMaskedModel)

    algo = build_config(num_players=num_players).build()
    for i in range(num_iters):
        result = algo.train()
        runners = result.get("env_runners", result)
        mean_return = runners.get("episode_return_mean", float("nan"))
        ep_len = runners.get("episode_len_mean", float("nan"))
        print(f"iter {i:4d}  mean_return={mean_return:+.3f}  ep_len={ep_len:.1f}")
        if (i + 1) % 25 == 0:
            ckpt = algo.save()
            print(f"  saved checkpoint -> {ckpt}")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
