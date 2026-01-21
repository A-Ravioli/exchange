#!/usr/bin/env python3
"""train trading agents with ppo self-play"""

import numpy as np
from multi_agent_env import MultiAgentExchangeEnv
import torch
import torch.nn as nn
from torch.distributions import Normal
import wandb
import os
from datetime import datetime

# minimal ppo implementation for self-play with wandb tracking and mps support

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


def train_self_play(n_agents=4, n_iterations=1000, steps_per_iter=500, use_wandb=True):
    """train agents with self-play using ppo"""
    
    # setup device (mps for apple silicon, cuda for nvidia, cpu fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using apple mps for acceleration 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("using cuda for acceleration 🚀")
    else:
        device = torch.device("cpu")
        print("using cpu (consider getting a gpu!)")
    
    # initialize wandb
    if use_wandb:
        wandb.init(
            project="exchange-rl-selfplay",
            config={
                "n_agents": n_agents,
                "n_iterations": n_iterations,
                "steps_per_iter": steps_per_iter,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "device": str(device),
                "policy_diversity": True  # new feature
            },
            name=f"selfplay_{n_agents}agents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    env = MultiAgentExchangeEnv(n_agents=n_agents, max_steps=steps_per_iter)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # create one policy per agent and move to device
    policies = [PolicyNetwork(obs_dim, act_dim).to(device) for _ in range(n_agents)]
    values = [ValueNetwork(obs_dim).to(device) for _ in range(n_agents)]
    
    policy_opts = [torch.optim.Adam(p.parameters(), lr=3e-4) for p in policies]
    value_opts = [torch.optim.Adam(v.parameters(), lr=3e-4) for v in values]
    
    # Policy pool for diversity (prevents monoculture)
    import copy
    policy_pool = []
    
    best_avg_pnl = -float('inf')
    checkpoint_dir = "checkpoints/rl_selfplay"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for iteration in range(n_iterations):
        # collect rollout
        obs_all, _ = env.reset(seed=iteration)
        
        trajectories = [[] for _ in range(n_agents)]
        episode_rewards = [0.0] * n_agents
        
        for step in range(steps_per_iter):
            # all agents act
            actions = {}
            for i in range(n_agents):
                obs_tensor = torch.FloatTensor(obs_all[i]).to(device)
                action, log_prob = policies[i].act(obs_tensor)
                actions[i] = action.cpu().detach().numpy()
                trajectories[i].append({
                    'obs': obs_all[i],
                    'action': action.cpu().detach(),
                    'log_prob': log_prob.cpu().detach()
                })
            
            # step env
            obs_all, rewards, dones, truncs, infos = env.step(actions)
            
            # store rewards
            for i in range(n_agents):
                trajectories[i][-1]['reward'] = rewards[i]
                episode_rewards[i] += rewards[i]
        
        # compute returns and advantages
        policy_losses = []
        value_losses = []
        
        for i in range(n_agents):
            returns = []
            G = 0
            for t in reversed(trajectories[i]):
                G = t['reward'] + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns).to(device)
            obs = torch.FloatTensor([t['obs'] for t in trajectories[i]]).to(device)
            
            with torch.no_grad():
                values_pred = values[i](obs)
            
            advantages = returns - values_pred
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # ppo update
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            
            for epoch in range(4):
                # policy update
                new_dist = policies[i](obs)
                new_log_probs = new_dist.log_prob(
                    torch.stack([t['action'].to(device) for t in trajectories[i]])
                ).sum(-1)
                
                old_log_probs = torch.stack([t['log_prob'].to(device) for t in trajectories[i]])
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
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
            
            policy_losses.append(epoch_policy_loss / 4)
            value_losses.append(epoch_value_loss / 4)
        
        # get final pnls
        final_pnls = [env.agents[i].cash for i in range(n_agents)]
        avg_pnl = np.mean(final_pnls)
        max_pnl = np.max(final_pnls)
        total_trades = sum(env.agents[i].trades for i in range(n_agents))
        
        # log to wandb
        if use_wandb:
            wandb.log({
                "iteration": iteration,
                "avg_pnl": avg_pnl,
                "max_pnl": max_pnl,
                "min_pnl": np.min(final_pnls),
                "avg_policy_loss": np.mean(policy_losses),
                "avg_value_loss": np.mean(value_losses),
                "total_trades": total_trades,
                **{f"agent_{i}_pnl": final_pnls[i] for i in range(n_agents)},
                **{f"agent_{i}_trades": env.agents[i].trades for i in range(n_agents)}
            })
        
        # log progress
        if iteration % 10 == 0:
            print(f"Iter {iteration}: avg_pnl={avg_pnl:.2f}, max_pnl={max_pnl:.2f}, trades={total_trades}")
        
        # Add diversity: save best policies to pool every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            best_agent_idx = np.argmax(final_pnls)
            policy_copy = copy.deepcopy(policies[best_agent_idx])
            policy_copy.eval()  # set to eval mode
            policy_pool.append(policy_copy)
            if len(policy_pool) > 10:  # keep last 10 policies
                policy_pool.pop(0)
            if use_wandb:
                wandb.log({"policy_pool_size": len(policy_pool)})
        
        # save checkpoint (only keep best)
        if avg_pnl > best_avg_pnl:
            best_avg_pnl = avg_pnl
            # delete old checkpoint
            for f in os.listdir(checkpoint_dir):
                if f.startswith("policies_"):
                    os.remove(os.path.join(checkpoint_dir, f))
            # save new checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"policies_iter_{iteration}_pnl_{avg_pnl:.2f}.pt")
            torch.save({
                'iteration': iteration,
                'policies': [p.state_dict() for p in policies],
                'values': [v.state_dict() for v in values],
                'avg_pnl': avg_pnl
            }, checkpoint_path)
            print(f"💾 saved checkpoint: {checkpoint_path}")
    
    if use_wandb:
        wandb.finish()
    
    return policies


if __name__ == "__main__":
    print("🚀 training agents with self-play ppo...")
    print("this will run for a LONG time. grab some coffee (or take a nap).")
    
    # LONG training run - adjust n_iterations to run longer
    policies = train_self_play(
        n_agents=4, 
        n_iterations=100000,  # 100k iterations
        steps_per_iter=500,
        use_wandb=True
    )
    
    print("✅ training complete!")
