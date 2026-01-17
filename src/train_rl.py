#!/usr/bin/env python3
"""train trading agents with ppo self-play"""

import numpy as np
from multi_agent_env import MultiAgentExchangeEnv
import torch
import torch.nn as nn
from torch.distributions import Normal

# minimal ppo implementation for self-play

class PolicyNetwork(nn.Module):
    """simple mlp policy"""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)
    
    def act(self, obs, deterministic=False):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class ValueNetwork(nn.Module):
    """value function"""
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def train_self_play(n_agents=4, n_iterations=1000, steps_per_iter=500):
    """train agents with self-play using ppo"""
    env = MultiAgentExchangeEnv(n_agents=n_agents, max_steps=steps_per_iter)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # create one policy per agent
    policies = [PolicyNetwork(obs_dim, act_dim) for _ in range(n_agents)]
    values = [ValueNetwork(obs_dim) for _ in range(n_agents)]
    
    policy_opts = [torch.optim.Adam(p.parameters(), lr=3e-4) for p in policies]
    value_opts = [torch.optim.Adam(v.parameters(), lr=3e-4) for v in values]
    
    for iteration in range(n_iterations):
        # collect rollout
        obs_all, _ = env.reset(seed=iteration)
        
        trajectories = [[] for _ in range(n_agents)]
        
        for step in range(steps_per_iter):
            # all agents act
            actions = {}
            for i in range(n_agents):
                obs_tensor = torch.FloatTensor(obs_all[i])
                action, log_prob = policies[i].act(obs_tensor)
                actions[i] = action.detach().numpy()
                trajectories[i].append({
                    'obs': obs_all[i],
                    'action': action.detach(),
                    'log_prob': log_prob.detach()
                })
            
            # step env
            obs_all, rewards, dones, truncs, infos = env.step(actions)
            
            # store rewards
            for i in range(n_agents):
                trajectories[i][-1]['reward'] = rewards[i]
        
        # compute returns and advantages
        for i in range(n_agents):
            returns = []
            G = 0
            for t in reversed(trajectories[i]):
                G = t['reward'] + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns)
            obs = torch.FloatTensor([t['obs'] for t in trajectories[i]])
            
            with torch.no_grad():
                values_pred = values[i](obs)
            
            advantages = returns - values_pred
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # ppo update
            for epoch in range(4):
                # policy update
                new_dist = policies[i](obs)
                new_log_probs = new_dist.log_prob(
                    torch.stack([t['action'] for t in trajectories[i]])
                ).sum(-1)
                
                old_log_probs = torch.stack([t['log_prob'] for t in trajectories[i]])
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages
                ).mean()
                
                policy_opts[i].zero_grad()
                policy_loss.backward()
                policy_opts[i].step()
                
                # value update
                value_pred = values[i](obs)
                value_loss = (value_pred - returns).pow(2).mean()
                
                value_opts[i].zero_grad()
                value_loss.backward()
                value_opts[i].step()
        
        # log progress
        if iteration % 10 == 0:
            final_pnls = [env.agents[i].cash for i in range(n_agents)]
            print(f"Iter {iteration}: PnLs = {[f'{p:.2f}' for p in final_pnls]}")
    
    return policies


if __name__ == "__main__":
    print("training agents with self-play ppo...")
    policies = train_self_play(n_agents=4, n_iterations=200, steps_per_iter=300)
    
    # save policies
    for i, policy in enumerate(policies):
        torch.save(policy.state_dict(), f"policy_agent_{i}.pt")
    
    print("training complete! policies saved.")
