import tianshou
from tianshou.data.buffer import ReplayBuffer
import numpy as np
import torch
from tianshou.data import Batch
from typing import Any, Tuple, Union, Optional


class HERReplayBuffer(ReplayBuffer):
    def __init__(self, size,
                 n_samples,
                 reward_fn,
                 sample_fn,
                 done_fn,
                 **kwargs):
        super().__init__(size, **kwargs)
        self.n_samples = n_samples
        self.reward_fn = reward_fn
        self.sample_fn = sample_fn
        self.done_fn = done_fn

    def update(self, buffer: 'ReplayBuffer') -> None:
        super().update(buffer)
        tmp_buffer = ReplayBuffer(size=len(buffer))
        for i in range(self.n_samples):
            for j in range(len(buffer)):
                batch = buffer[j]
                sampled_indice = self.sample_fn(buffer, j)
                sampled_batch = buffer[sampled_indice]
                # rewrite the goal
                batch.obs.desired_goal = sampled_batch.obs_next.achieved_goal
                batch.obs_next.desired_goal = sampled_batch.obs_next.achieved_goal
                # update the reward
                batch.rew = self.reward_fn(batch.obs_next.achieved_goal, batch.obs_next.desired_goal)
                # update the done flag
                batch.done = self.done_fn(batch.obs, batch.act, batch.obs_next, batch.rew)
                tmp_buffer.add(**batch)
            super().update(tmp_buffer)
            tmp_buffer.reset()
