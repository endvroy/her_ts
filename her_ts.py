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
    return BitFlipEnv(n_bits, continuous=True)


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
        outp = 2 * torch.sigmoid(raw_outp) - 1
        return outp, state


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super().__init__()
        self.model = build_nn(state_dim + action_dim, hidden_dims, 1)

    def forward(self, obs, raw_action):
        observation = torch.from_numpy(obs.observation).float()
        desired_goal = torch.from_numpy(obs.desired_goal).float()
        action = torch.Tensor(raw_action).float()
        inp = torch.cat([observation, desired_goal, action], dim=1)
        value = self.model(inp)
        return value


class BitFlipModule:
    def __init__(self,
                 n_bits,
                 n_train_envs,
                 n_test_envs,
                 use_her):
        self.env = make_env(n_bits)
        train_envs = ts.env.DummyVectorEnv([lambda: make_env(n_bits) for _ in range(n_train_envs)])
        test_envs = ts.env.DummyVectorEnv([lambda: make_env(n_bits) for _ in range(n_test_envs)])
        state_dim = 2 * n_bits
        action_dim = n_bits
        actor = ActorNet(state_dim, [100, 200, 100], action_dim)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
        critic_1 = CriticNet(state_dim, [100, 200, 100], action_dim)
        critic_1_optim = torch.optim.Adam(critic_1.parameters(), lr=1e-3)
        critic_2 = CriticNet(state_dim, [100, 200, 100], action_dim)
        critic_2_optim = torch.optim.Adam(critic_2.parameters(), lr=1e-3)

        self.policy = ts.policy.TD3Policy(actor=actor,
                                          actor_optim=actor_optim,
                                          critic1=critic_1,
                                          critic1_optim=critic_1_optim,
                                          critic2=critic_2,
                                          critic2_optim=critic_2_optim,
                                          gamma=1,
                                          estimation_step=3,
                                          action_range=(-1, 1))

        def future_sampling(buffer, index):
            rng = np.random.default_rng()
            sampled_index = rng.integers(index, len(buffer))
            return sampled_index

        def done_fn(obs, action, obs_next, reward):
            return reward == 1

        if use_her:
            buffer = HERReplayBuffer(size=20000,
                                     n_samples=2,
                                     reward_fn=functools.partial(self.env.compute_reward, _info=None),
                                     sample_fn=future_sampling,
                                     done_fn=done_fn)
        else:
            buffer = ts.data.ReplayBuffer(20000)

        self.train_collector = ts.data.Collector(self.policy, train_envs, buffer)
        self.test_collector = ts.data.Collector(self.policy, test_envs)

    def train(self):
        writer = SummaryWriter('log/bitflip')
        result = ts.trainer.offpolicy_trainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            max_epoch=20,
            step_per_epoch=1000,
            collect_per_step=10,
            episode_per_test=100,
            batch_size=64,
            # train_fn=lambda e: self.policy.set_eps(0.1),
            # test_fn=lambda e: self.policy.set_eps(0.05),
            # stop_fn=lambda x: x >= 0.7,
            writer=writer)
        return result

    def eval(self):
        self.policy.eval()
        collector = ts.data.Collector(self.policy, self.env)
        collector.collect(n_episode=1, render=1 / 35)


if __name__ == '__main__':
    use_her = True
    bitflip_module = BitFlipModule(n_bits=6,
                                   n_train_envs=8,
                                   n_test_envs=100,
                                   use_her=use_her)
    print(f'use_her={use_her}')
    bitflip_module.train()
