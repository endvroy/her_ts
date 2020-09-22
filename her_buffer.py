import tianshou
from tianshou.data.buffer import ReplayBuffer
import numpy as np
import torch
from tianshou.data import Batch
from typing import Any, Tuple, Union, Optional


class HERReplayBuffer(ReplayBuffer):
    def __init__(self, size, n_samples, reward_fn, **kwargs):
        super().__init__(size, **kwargs)
        self.n_samples = n_samples
        self.reward_fn = reward_fn

    def update(self, buffer: 'ReplayBuffer') -> None:
        super().update(buffer)
        tmp_buffer = ReplayBuffer(size=len(buffer))
        rng = np.random.default_rng()
        for i in range(self.n_samples):
            for j in range(len(buffer)):
                batch = buffer[j]
                sampled_indice = len(buffer) - 1  # final sampling
                sampled_batch = buffer[sampled_indice]
                # rewrite the goal
                batch.obs.desired_goal = sampled_batch.obs_next.achieved_goal
                batch.obs_next.desired_goal = sampled_batch.obs_next.achieved_goal
                # update the reward
                batch.rew = self.reward_fn(batch.obs_next.achieved_goal, batch.obs_next.desired_goal)
                # update the done flag
                # todo: change to be more general
                batch.done = (batch.rew == 1)
                tmp_buffer.add(**batch)
            super().update(tmp_buffer)
            tmp_buffer.reset()
