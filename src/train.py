from typing import Dict
import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

from envs.env import PassageEnv, PassageGymEnv

from models.model import Model
from ray.rllib.models import ModelCatalog

from rllib_multi_agent_demo.multi_trainer import MultiPPOTrainer
from rllib_multi_agent_demo.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)


def initialize():
    ray.init(
        _temp_dir="/tmp/test",
        # local_mode=True,
        dashboard_host="0.0.0.0",
        object_store_memory=8 * (10 ** 9),
    )

    register_env("passage_env", lambda config: PassageGymEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )

        episode.user_data["makespan"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        info = episode.last_info_for()
        # covered_targets = np.sum(list(info["n_covered_targets"].values()))
        # episode.user_data["covered_targets"].append(covered_targets)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        pass
        # covered_targets = np.sum(episode.user_data["covered_targets"])
        # episode.custom_metrics["covered_targets"] = covered_targets


def train():
    tune.run(
        MultiPPOTrainer,
        # restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        # checkpoint_score_attr="min-episode_len_mean",
        checkpoint_score_attr="custom_metrics/covered_targets_mean",
        # local_dir="./results",
        local_dir="/tmp",
        # loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config={
            "seed": 0,
            "framework": "torch",
            "env": "passage_env",
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "train_batch_size": 1000,  # 65536,
            "sgd_minibatch_size": 256,  # 4096,
            "vf_clip_param": 100.0,
            "num_sgd_iter": 8,  # 18,
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "lr": 5e-5,
            "gamma": 0.995,
            "batch_mode": "truncate_episodes",  # complete_episodes, truncate_episodes
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "model",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "activation": "relu",
                    "msg_features": 16,
                    "comm_range": 2.0,
                },
            },
            "logger_config": {
                "wandb": {
                    "project": "marl_object_tracking",
                    "group": "global",
                    "api_key_file": "./src/wandb_api_key_file",
                }
            },
            "env_config": {
                "world_dim": (4.0, 6.0),
                "dt": 0.05,
                "num_envs": 8,
                "device": "cpu",
                "n_agents": 5,
                "agent_formation": (
                    np.array([[-1, -1], [-1, 1], [0, 0], [1, -1], [1, 1]]) * 0.6
                ).tolist(),
                "placement_keepout_border": 1.0,
                "placement_keepout_wall": 1.5,
                "pos_noise_std": 0.0,
                "max_time_steps": 500,
                "communication_range": 2.0,
                "wall_width": 0.3,
                "gap_length": 1.0,
                "grid_px_per_m": 40,
                "agent_radius": 0.25,
                "render": False,
                "render_px_per_m": 40,
                "max_v": 1.0,
                "max_a": 1.0,
            },
            "render_env": False,
            "evaluation_interval": 50,
            "evaluation_num_episodes": 1,
            "evaluation_num_workers": 1,  # Run evaluation in parallel to training
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "record_env": "videos",
                "render_env": True,
            },
            "callbacks": MyCallbacks,
        },
    )


if __name__ == "__main__":
    initialize()
    train()
