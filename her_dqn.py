import math
import functools
import torch
from torch import nn
import torch.nn.functional as F
from bit_env import BitFlipEnv
import tianshou as ts
# from tianshou.utils.net.continuous import Actor, Critic
from tianshou.data import Batch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from her_buffer import HERReplayBuffer


def make_env(n_bits):
    return BitFlipEnv(n_bits, continuous=False)


def build_nn(inp_dim, hidden_dims, out_dim):
    dims = [inp_dim] + hidden_dims + [out_dim]
    nets = []
    for i in range(len(dims) - 1):
        nets.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            nets.append(nn.ReLU())

    # nets.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*nets)


class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super().__init__()
        self.model = build_nn(state_dim, hidden_dims, action_dim)

    def forward(self, obs, state=None, info=None):
        observation = torch.from_numpy(obs.observation).float()
        desired_goal = torch.from_numpy(obs.desired_goal).float()
        inp = torch.cat([observation, desired_goal], dim=1)
        raw_outp = self.model(inp)
        # outp = torch.sigmoid(raw_outp)
        outp = raw_outp
        return outp, state


class BitFlipModule:
    def __init__(self,
                 n_bits,
                 n_train_envs,
                 n_test_envs,
                 buffer_size,
                 use_her):
        self.env = make_env(n_bits)
        train_envs = ts.env.DummyVectorEnv([lambda: make_env(n_bits) for _ in range(n_train_envs)])
        test_envs = ts.env.DummyVectorEnv([lambda: make_env(n_bits) for _ in range(n_test_envs)])
        state_dim = self.env.observation_space['observation'].n + self.env.observation_space['desired_goal'].n
        action_dim = self.env.action_space.n
        actor = ActorNet(state_dim, [100, 200, 100], action_dim)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=0.001)

        self.policy = ts.policy.DQNPolicy(model=actor,
                                          optim=actor_optim,
                                          discount_factor=0.95,
                                          estimation_step=1,
                                          target_update_freq=320)

        def future_sampling(buffer, index):
            rng = np.random.default_rng()
            sampled_index = rng.integers(index, len(buffer))
            return sampled_index

        def done_fn(obs, action, obs_next, reward):
            return reward == 1

        if use_her:
            buffer = HERReplayBuffer(size=buffer_size,
                                     n_samples=4,
                                     reward_fn=functools.partial(self.env.compute_reward, _info=None),
                                     sample_fn=future_sampling,
                                     done_fn=done_fn)
        else:
            buffer = ts.data.ReplayBuffer(size=buffer_size)

        self.train_collector = ts.data.Collector(self.policy, train_envs, buffer)
        self.test_collector = ts.data.Collector(self.policy, test_envs)

    def train(self):
        writer = SummaryWriter('log/bitflip')
        result = ts.trainer.offpolicy_trainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            max_epoch=200,
            step_per_epoch=40,
            collect_per_step=16,
            episode_per_test=100,
            batch_size=128,
            train_fn=lambda e: self.policy.set_eps(0.1),
            test_fn=lambda e: self.policy.set_eps(0.05),
            # stop_fn=lambda x: x >= 0.7,
            writer=writer)
        return result

    def eval(self):
        self.policy.eval()
        collector = ts.data.Collector(self.policy, self.env)
        collector.collect(n_episode=1, render=1 / 35)


if __name__ == '__main__':
    use_her = True
    bitflip_module = BitFlipModule(n_bits=20,
                                   n_train_envs=8,
                                   n_test_envs=100,
                                   buffer_size=int(1e6),
                                   use_her=use_her)
    print(f'use_her={use_her}')
    bitflip_module.train()
