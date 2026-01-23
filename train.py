#!/usr/bin/env python3
# consolidated training script for exchange rl agents

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import os
import copy
# import from src
from src.parallel_env import ParallelEnv
from src.multi_agent_env import MultiAgentExchangeEnv
from src.networks import LargePolicyNetwork, LargeValueNetwork, PolicyNetwork, ValueNetwork
from src.evolve import RuleBasedAgent, evolve_strategies

import wandb


def train_rl(
    n_agents=4,
    n_iterations=1000,
    steps_per_iter=500,
    n_envs=32,
    use_wandb=True,
    network_size='large',
    ppo_epochs=20,
    use_mixed_precision=True,
    mini_batch_size=256,
    mode='rl'  # 'rl', 'evolution', or 'hybrid'
):
    """
    unified training function with all optimizations
    - parallel environments
    - larger networks
    - mixed precision training
    - mini-batch ppo
    """
    
    # setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using apple mps for acceleration 🚀")
        use_mixed_precision = False  # mps doesn't support amp yet
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("using cuda for acceleration 🚀")
    else:
        device = torch.device("cpu")
        print("using cpu")
        use_mixed_precision = False
    
    # initialize wandb
    if use_wandb:
        wandb.init(
            project=f"exchange-{mode}",
            config={
                "n_agents": n_agents,
                "n_iterations": n_iterations,
                "steps_per_iter": steps_per_iter,
                "n_envs": n_envs,
                "ppo_epochs": ppo_epochs,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "device": str(device),
                "network_size": network_size,
                "mixed_precision": use_mixed_precision,
                "mode": mode
            },
            name=f"{mode}_{n_envs}envs_{network_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # create parallel environments
    env_fns = [
        lambda: MultiAgentExchangeEnv(n_agents=n_agents, max_steps=steps_per_iter)
        for _ in range(n_envs)
    ]
    par_env = ParallelEnv(env_fns, n_envs=n_envs)
    
    obs_dim = par_env.observation_space.shape[0]
    act_dim = par_env.action_space.shape[0]
    
    # create networks
    if network_size == 'large':
        policies = [LargePolicyNetwork(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        values = [LargeValueNetwork(obs_dim).to(device) for _ in range(n_agents)]
    else:
        policies = [PolicyNetwork(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        values = [ValueNetwork(obs_dim).to(device) for _ in range(n_agents)]
    
    policy_opts = [torch.optim.Adam(p.parameters(), lr=3e-4) for p in policies]
    value_opts = [torch.optim.Adam(v.parameters(), lr=3e-4) for v in values]
    
    # mixed precision scaler
    scaler = GradScaler() if use_mixed_precision else None
    
    # policy pool for diversity
    policy_pool = []
    
    best_avg_pnl = -float('inf')
    checkpoint_dir = f"checkpoints/{mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n🚀 starting {mode} training:")
    print(f"   {n_envs} parallel environments")
    print(f"   {network_size} networks")
    print(f"   {ppo_epochs} ppo epochs")
    print(f"   mixed precision: {use_mixed_precision}\n")
    
    for iteration in range(n_iterations):
        # collect rollouts from all parallel environments
        obs_all = par_env.reset(seed=iteration)
        
        # trajectories for each agent, across all envs
        trajectories = [[[] for _ in range(n_envs)] for _ in range(n_agents)]
        episode_rewards = [[0.0 for _ in range(n_envs)] for _ in range(n_agents)]
        
        for step in range(steps_per_iter):
            actions_per_agent = {}
            
            for agent_id in range(n_agents):
                obs_tensor = torch.FloatTensor(obs_all[agent_id]).to(device)
                
                with torch.no_grad():
                    if use_mixed_precision:
                        with autocast():
                            action, log_prob = policies[agent_id].act(obs_tensor)
                    else:
                        action, log_prob = policies[agent_id].act(obs_tensor)
                
                actions_per_agent[agent_id] = action.cpu().numpy()
                
                for env_id in range(n_envs):
                    trajectories[agent_id][env_id].append({
                        'obs': obs_all[agent_id][env_id],
                        'action': action[env_id].cpu(),
                        'log_prob': log_prob[env_id].cpu()
                    })
            
            # convert to list of dicts (one dict per env)
            actions_list = []
            for env_id in range(n_envs):
                env_actions = {
                    agent_id: actions_per_agent[agent_id][env_id]
                    for agent_id in range(n_agents)
                }
                actions_list.append(env_actions)
            
            obs_all, rewards, dones, truncs, infos = par_env.step(actions_list)
            
            for agent_id in range(n_agents):
                for env_id in range(n_envs):
                    trajectories[agent_id][env_id][-1]['reward'] = rewards[agent_id][env_id]
                    episode_rewards[agent_id][env_id] += rewards[agent_id][env_id]
        
        # training loop with mini-batch ppo
        policy_losses = []
        value_losses = []
        
        for agent_id in range(n_agents):
            # combine trajectories from all envs
            all_obs = []
            all_actions = []
            all_log_probs = []
            all_returns = []
            
            for env_id in range(n_envs):
                traj = trajectories[agent_id][env_id]
                
                returns = []
                G = 0
                for t in reversed(traj):
                    G = t['reward'] + 0.99 * G
                    returns.insert(0, G)
                
                all_obs.extend([t['obs'] for t in traj])
                all_actions.extend([t['action'] for t in traj])
                all_log_probs.extend([t['log_prob'] for t in traj])
                all_returns.extend(returns)
            
            obs_batch = torch.FloatTensor(np.array(all_obs)).to(device)
            actions_batch = torch.stack(all_actions).to(device)
            old_log_probs_batch = torch.stack(all_log_probs).to(device)
            returns_batch = torch.FloatTensor(all_returns).to(device)
            
            with torch.no_grad():
                values_pred = values[agent_id](obs_batch)
            
            advantages = returns_batch - values_pred
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # ppo epochs with mini-batches
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            
            for epoch in range(ppo_epochs):
                # mini-batch training
                n_samples = len(obs_batch)
                indices = torch.randperm(n_samples)
                
                for start_idx in range(0, n_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, n_samples)
                    mb_indices = indices[start_idx:end_idx]
                    
                    mb_obs = obs_batch[mb_indices]
                    mb_actions = actions_batch[mb_indices]
                    mb_old_log_probs = old_log_probs_batch[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns_batch[mb_indices]
                    
                    if use_mixed_precision:
                        with autocast():
                            # policy loss
                            new_dist = policies[agent_id](mb_obs)
                            new_log_probs = new_dist.log_prob(mb_actions).sum(-1)
                            ratio = torch.exp(new_log_probs - mb_old_log_probs)
                            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                            policy_loss = -torch.min(
                                ratio * mb_advantages,
                                clipped_ratio * mb_advantages
                            ).mean()
                            
                            # value loss
                            value_pred = values[agent_id](mb_obs)
                            value_loss = (value_pred - mb_returns).pow(2).mean()
                        
                        # backward with scaling
                        scaler.scale(policy_loss).backward()
                        scaler.step(policy_opts[agent_id])
                        scaler.update()
                        policy_opts[agent_id].zero_grad()
                        
                        scaler.scale(value_loss).backward()
                        scaler.step(value_opts[agent_id])
                        scaler.update()
                        value_opts[agent_id].zero_grad()
                    else:
                        # standard precision
                        new_dist = policies[agent_id](mb_obs)
                        new_log_probs = new_dist.log_prob(mb_actions).sum(-1)
                        ratio = torch.exp(new_log_probs - mb_old_log_probs)
                        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                        policy_loss = -torch.min(
                            ratio * mb_advantages,
                            clipped_ratio * mb_advantages
                        ).mean()
                        
                        policy_opts[agent_id].zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(policies[agent_id].parameters(), 0.5)
                        policy_opts[agent_id].step()
                        
                        value_pred = values[agent_id](mb_obs)
                        value_loss = (value_pred - mb_returns).pow(2).mean()
                        
                        value_opts[agent_id].zero_grad()
                        value_loss.backward()
                        torch.nn.utils.clip_grad_norm_(values[agent_id].parameters(), 0.5)
                        value_opts[agent_id].step()
                    
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
            
            policy_losses.append(epoch_policy_loss / (ppo_epochs * (n_samples // mini_batch_size + 1)))
            value_losses.append(epoch_value_loss / (ppo_epochs * (n_samples // mini_batch_size + 1)))
        
        # get final pnls
        final_pnls = [np.mean(episode_rewards[i]) for i in range(n_agents)]
        avg_pnl = np.mean(final_pnls)
        max_pnl = np.max(final_pnls)
        
        # log to wandb
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
        
        # log progress
        if iteration % 10 == 0:
            print(f"iter {iteration}: avg_pnl={avg_pnl:.2f}, max_pnl={max_pnl:.2f}")
        
        # policy diversity
        if iteration > 0 and iteration % 50 == 0:
            best_agent_idx = np.argmax(final_pnls)
            policy_copy = copy.deepcopy(policies[best_agent_idx])
            policy_copy.eval()
            policy_pool.append(policy_copy)
            if len(policy_pool) > 10:
                policy_pool.pop(0)
        
        # save checkpoint
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
                'config': {
                    'n_envs': n_envs,
                    'network_size': network_size,
                    'ppo_epochs': ppo_epochs,
                    'mixed_precision': use_mixed_precision
                }
            }, checkpoint_path)
            print(f"💾 saved checkpoint: {checkpoint_path}")
    
    par_env.close()
    
    if use_wandb:
        wandb.finish()
    
    return policies


def main():
    parser = argparse.ArgumentParser(description='train exchange trading agents')
    parser.add_argument('--mode', type=str, default='rl', choices=['rl', 'evolution', 'hybrid'],
                        help='training mode (default: rl)')
    parser.add_argument('--n_agents', type=int, default=4, help='number of agents')
    parser.add_argument('--n_iterations', type=int, default=1000, help='training iterations')
    parser.add_argument('--steps_per_iter', type=int, default=500, help='steps per iteration')
    parser.add_argument('--n_envs', type=int, default=32, help='parallel environments')
    parser.add_argument('--network_size', type=str, default='large', choices=['small', 'large'],
                        help='network size')
    parser.add_argument('--ppo_epochs', type=int, default=20, help='ppo epochs')
    parser.add_argument('--mini_batch_size', type=int, default=256, help='mini batch size')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb logging')
    parser.add_argument('--no_mixed_precision', action='store_true', help='disable mixed precision')
    
    args = parser.parse_args()
    
    print(f"🚀 starting {args.mode} training...")
    print(f"   agents: {args.n_agents}")
    print(f"   iterations: {args.n_iterations}")
    print(f"   parallel envs: {args.n_envs}")
    print(f"   network: {args.network_size}")
    print()
    
    if args.mode == 'evolution':
        # run evolution instead of rl
        env = MultiAgentExchangeEnv(n_agents=args.n_agents, max_steps=args.steps_per_iter)
        population = evolve_strategies(
            pop_size=args.n_agents * 4,
            n_generations=args.n_iterations,
            use_wandb=not args.no_wandb
        )
        print("\n✅ evolution complete!")
    else:
        policies = train_rl(
            n_agents=args.n_agents,
            n_iterations=args.n_iterations,
            steps_per_iter=args.steps_per_iter,
            n_envs=args.n_envs,
            use_wandb=not args.no_wandb,
            network_size=args.network_size,
            ppo_epochs=args.ppo_epochs,
            use_mixed_precision=not args.no_mixed_precision,
            mini_batch_size=args.mini_batch_size,
            mode=args.mode
        )
        print("\n✅ rl training complete!")


if __name__ == "__main__":
    main()
