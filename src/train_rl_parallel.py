#!/usr/bin/env python3
"""
RL training with parallel environments and larger networks.
Combines Phase 1 (parallel envs) and Phase 2 (larger networks).
"""

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import os
import copy

# Import new components
from parallel_env import ParallelEnv
from multi_agent_env import MultiAgentExchangeEnv
from networks import LargePolicyNetwork, LargeValueNetwork

import wandb


def train_self_play_parallel(
    n_agents=4, 
    n_iterations=1000, 
    steps_per_iter=500, 
    n_envs=8,  # NEW: parallel environments
    use_wandb=True,
    network_size='large'  # 'large' or 'xlarge'
):
    """
    Train agents with self-play using PPO with parallel environments.
    
    Args:
        n_agents: Number of competing agents
        n_iterations: Training iterations
        steps_per_iter: Steps per iteration
        n_envs: Number of parallel environments
        use_wandb: Whether to log to wandb
        network_size: 'large' (512x3) or 'xlarge' (1024x4)
    """
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using apple mps for acceleration 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("using cuda for acceleration 🚀")
    else:
        device = torch.device("cpu")
        print("using cpu (consider getting a gpu!)")
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="exchange-rl-parallel",
            config={
                "n_agents": n_agents,
                "n_iterations": n_iterations,
                "steps_per_iter": steps_per_iter,
                "n_envs": n_envs,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "device": str(device),
                "policy_diversity": True,
                "network_size": network_size,
                "parallel_envs": True
            },
            name=f"parallel_{n_envs}envs_{network_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create parallel environments
    env_fns = [
        lambda: MultiAgentExchangeEnv(n_agents=n_agents, max_steps=steps_per_iter)
        for _ in range(n_envs)
    ]
    par_env = ParallelEnv(env_fns, n_envs=n_envs)
    
    obs_dim = par_env.observation_space.shape[0]
    act_dim = par_env.action_space.shape[0]
    
    # Create larger networks
    if network_size == 'xlarge':
        from networks import ExtraLargePolicyNetwork, ExtraLargeValueNetwork
        PolicyNet = ExtraLargePolicyNetwork
        ValueNet = ExtraLargeValueNetwork
    else:
        PolicyNet = LargePolicyNetwork
        ValueNet = LargeValueNetwork
    
    policies = [PolicyNet(obs_dim, act_dim).to(device) for _ in range(n_agents)]
    values = [ValueNet(obs_dim).to(device) for _ in range(n_agents)]
    
    policy_opts = [torch.optim.Adam(p.parameters(), lr=3e-4) for p in policies]
    value_opts = [torch.optim.Adam(v.parameters(), lr=3e-4) for v in values]
    
    # Policy pool for diversity
    policy_pool = []
    
    best_avg_pnl = -float('inf')
    checkpoint_dir = "checkpoints/rl_parallel"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for iteration in range(n_iterations):
        # Collect rollouts from ALL parallel environments
        obs_all = par_env.reset(seed=iteration)
        
        # Trajectories for each agent, across all envs
        trajectories = [[[] for _ in range(n_envs)] for _ in range(n_agents)]
        episode_rewards = [[0.0 for _ in range(n_envs)] for _ in range(n_agents)]
        
        for step in range(steps_per_iter):
            # All agents act across all envs
            actions_per_agent = {}
            for agent_id in range(n_agents):
                # obs_all[agent_id] has shape (n_envs, obs_dim)
                obs_tensor = torch.FloatTensor(obs_all[agent_id]).to(device)
                
                with torch.no_grad():
                    action, log_prob = policies[agent_id].act(obs_tensor)
                
                actions_per_agent[agent_id] = action.cpu().numpy()
                
                # Store trajectories for each env separately
                for env_id in range(n_envs):
                    trajectories[agent_id][env_id].append({
                        'obs': obs_all[agent_id][env_id],
                        'action': action[env_id].cpu(),
                        'log_prob': log_prob[env_id].cpu()
                    })
            
            # Convert to list of dicts (one dict per env)
            actions_list = []
            for env_id in range(n_envs):
                env_actions = {
                    agent_id: actions_per_agent[agent_id][env_id]
                    for agent_id in range(n_agents)
                }
                actions_list.append(env_actions)
            
            # Step all environments
            obs_all, rewards, dones, truncs, infos = par_env.step(actions_list)
            
            # Store rewards for each env
            for agent_id in range(n_agents):
                for env_id in range(n_envs):
                    trajectories[agent_id][env_id][-1]['reward'] = rewards[agent_id][env_id]
                    episode_rewards[agent_id][env_id] += rewards[agent_id][env_id]
        
        # Update each agent using data from all envs
        policy_losses = []
        value_losses = []
        
        for agent_id in range(n_agents):
            # Combine trajectories from all envs
            all_obs = []
            all_actions = []
            all_log_probs = []
            all_returns = []
            
            for env_id in range(n_envs):
                traj = trajectories[agent_id][env_id]
                
                # Compute returns for this env
                returns = []
                G = 0
                for t in reversed(traj):
                    G = t['reward'] + 0.99 * G
                    returns.insert(0, G)
                
                # Collect data
                all_obs.extend([t['obs'] for t in traj])
                all_actions.extend([t['action'] for t in traj])
                all_log_probs.extend([t['log_prob'] for t in traj])
                all_returns.extend(returns)
            
            # Convert to tensors
            obs_batch = torch.FloatTensor(np.array(all_obs)).to(device)
            actions_batch = torch.stack(all_actions).to(device)
            old_log_probs_batch = torch.stack(all_log_probs).to(device)
            returns_batch = torch.FloatTensor(all_returns).to(device)
            
            # Compute advantages
            with torch.no_grad():
                values_pred = values[agent_id](obs_batch)
            
            advantages = returns_batch - values_pred
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update (4 epochs)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            
            for epoch in range(4):
                # Policy update
                new_dist = policies[agent_id](obs_batch)
                new_log_probs = new_dist.log_prob(actions_batch).sum(-1)
                
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages
                ).mean()
                
                policy_opts[agent_id].zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policies[agent_id].parameters(), 0.5)
                policy_opts[agent_id].step()
                
                # Value update
                value_pred = values[agent_id](obs_batch)
                value_loss = (value_pred - returns_batch).pow(2).mean()
                
                value_opts[agent_id].zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(values[agent_id].parameters(), 0.5)
                value_opts[agent_id].step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
            
            policy_losses.append(epoch_policy_loss / 4)
            value_losses.append(epoch_value_loss / 4)
        
        # Get final PnLs (average across all envs)
        final_pnls = [np.mean(episode_rewards[i]) for i in range(n_agents)]
        avg_pnl = np.mean(final_pnls)
        max_pnl = np.max(final_pnls)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "iteration": iteration,
                "avg_pnl": avg_pnl,
                "max_pnl": max_pnl,
                "min_pnl": np.min(final_pnls),
                "avg_policy_loss": np.mean(policy_losses),
                "avg_value_loss": np.mean(value_losses),
                **{f"agent_{i}_pnl": final_pnls[i] for i in range(n_agents)}
            })
        
        # Log progress
        if iteration % 10 == 0:
            print(f"Iter {iteration}: avg_pnl={avg_pnl:.2f}, max_pnl={max_pnl:.2f}")
        
        # Add diversity: save best policies to pool
        if iteration > 0 and iteration % 50 == 0:
            best_agent_idx = np.argmax(final_pnls)
            policy_copy = copy.deepcopy(policies[best_agent_idx])
            policy_copy.eval()
            policy_pool.append(policy_copy)
            if len(policy_pool) > 10:
                policy_pool.pop(0)
            if use_wandb:
                wandb.log({"policy_pool_size": len(policy_pool)})
        
        # Save checkpoint (only keep best)
        if avg_pnl > best_avg_pnl:
            best_avg_pnl = avg_pnl
            for f in os.listdir(checkpoint_dir):
                if f.startswith("policies_"):
                    os.remove(os.path.join(checkpoint_dir, f))
            
            checkpoint_path = os.path.join(checkpoint_dir, f"policies_iter_{iteration}_pnl_{avg_pnl:.2f}.pt")
            torch.save({
                'iteration': iteration,
                'policies': [p.state_dict() for p in policies],
                'values': [v.state_dict() for v in values],
                'avg_pnl': avg_pnl,
                'n_envs': n_envs,
                'network_size': network_size
            }, checkpoint_path)
            print(f"💾 saved checkpoint: {checkpoint_path}")
    
    # Cleanup
    par_env.close()
    
    if use_wandb:
        wandb.finish()
    
    return policies


if __name__ == "__main__":
    print("🚀 Training agents with parallel envs and large networks...")
    print("This uses Phase 1 + Phase 2 optimizations.")
    
    policies = train_self_play_parallel(
        n_agents=4,
        n_iterations=100,  # Short run for testing
        steps_per_iter=100,
        n_envs=8,
        use_wandb=False,  # Set to True for tracking
        network_size='large'
    )
    
    print("✅ Training complete!")
