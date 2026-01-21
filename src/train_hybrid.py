#!/usr/bin/env python3
"""hybrid training: evolved strategies compete with rl agents"""

import numpy as np
import torch
import torch.nn as nn
from multi_agent_env import MultiAgentExchangeEnv
from evolve import RuleBasedAgent
from train_rl import PolicyNetwork
import wandb
import os
from datetime import datetime

# hybrid evolution + rl with wandb tracking

def train_hybrid(n_rule_agents=2, n_rl_agents=2, n_iterations=10000, use_wandb=True):
    """train rl agents against evolved rule-based agents"""
    
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
    
    n_total = n_rule_agents + n_rl_agents
    
    if use_wandb:
        wandb.init(
            project="exchange-hybrid",
            config={
                "n_rule_agents": n_rule_agents,
                "n_rl_agents": n_rl_agents,
                "n_iterations": n_iterations,
                "device": str(device)
            },
            name=f"hybrid_{n_rule_agents}rule_{n_rl_agents}rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    env = MultiAgentExchangeEnv(n_agents=n_total, max_steps=500)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # initialize rule-based agents with diverse starting parameters
    rule_agents = []
    for i in range(n_rule_agents):
        # Start with different initialization strategies
        agent = RuleBasedAgent()
        if i > 0:  # mutate the first one to create diversity
            agent = agent.mutate(0.5)
        rule_agents.append(agent)
    
    # initialize rl agents
    rl_policies = [PolicyNetwork(obs_dim, act_dim).to(device) for _ in range(n_rl_agents)]
    rl_values = [nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    ).to(device) for _ in range(n_rl_agents)]
    
    rl_opts = [torch.optim.Adam(list(rl_policies[i].parameters()) + list(rl_values[i].parameters()), 
                                lr=3e-4) for i in range(n_rl_agents)]
    
    checkpoint_dir = "checkpoints/hybrid"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_rl_pnl = -float('inf')
    
    # evolve rule agents every N iterations
    evolve_interval = 100
    
    for iteration in range(n_iterations):
        obs_all, _ = env.reset(seed=iteration)
        
        # trajectories for rl agents only
        rl_trajectories = [[] for _ in range(n_rl_agents)]
        
        done = False
        step_count = 0
        
        while not done and step_count < 500:
            actions = {}
            
            # rule agents act
            for i in range(n_rule_agents):
                actions[i] = rule_agents[i].get_action(obs_all[i])
            
            # rl agents act and record
            for i in range(n_rl_agents):
                rl_idx = n_rule_agents + i
                obs_tensor = torch.FloatTensor(obs_all[rl_idx]).to(device)
                action, log_prob = rl_policies[i].act(obs_tensor)
                actions[rl_idx] = action.cpu().detach().numpy()
                rl_trajectories[i].append({
                    'obs': obs_all[rl_idx],
                    'action': action.cpu().detach(),
                    'log_prob': log_prob.cpu().detach(),
                })
            
            obs_all, rewards, dones, _, infos = env.step(actions)
            
            # store rewards for rl agents
            for i in range(n_rl_agents):
                rl_idx = n_rule_agents + i
                rl_trajectories[i][-1]['reward'] = rewards[rl_idx]
            
            done = dones[0]
            step_count += 1
        
        # update rl agents with ppo
        for i in range(n_rl_agents):
            if len(rl_trajectories[i]) == 0:
                continue
            
            returns = []
            G = 0
            for t in reversed(rl_trajectories[i]):
                G = t['reward'] + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns).to(device)
            obs = torch.FloatTensor([t['obs'] for t in rl_trajectories[i]]).to(device)
            actions = torch.stack([t['action'].to(device) for t in rl_trajectories[i]])
            old_log_probs = torch.stack([t['log_prob'].to(device) for t in rl_trajectories[i]])
            
            with torch.no_grad():
                values_pred = rl_values[i](obs).squeeze()
            
            advantages = returns - values_pred
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # ppo epochs
            for _ in range(4):
                new_dist = rl_policies[i](obs)
                new_log_probs = new_dist.log_prob(actions).sum(-1)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped = torch.clamp(ratio, 0.8, 1.2)
                
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                value_loss = (rl_values[i](obs).squeeze() - returns).pow(2).mean()
                
                loss = policy_loss + 0.5 * value_loss
                
                rl_opts[i].zero_grad()
                loss.backward()
                rl_opts[i].step()
        
        # periodically evolve the rule agents (make them harder)
        if iteration > 0 and iteration % evolve_interval == 0:
            rule_pnls = [infos[i]["pnl"] for i in range(n_rule_agents)]
            best_rule_idx = np.argmax(rule_pnls)
            
            # mutate best rule agent and replace worst
            worst_rule_idx = np.argmin(rule_pnls)
            rule_agents[worst_rule_idx] = rule_agents[best_rule_idx].mutate(0.1)
            print(f"🧬 evolved rule agents at iter {iteration}")
        
        # logging
        rule_pnls = [infos[i]["pnl"] for i in range(n_rule_agents)]
        rl_pnls = [infos[n_rule_agents + i]["pnl"] for i in range(n_rl_agents)]
        
        avg_rule_pnl = np.mean(rule_pnls)
        avg_rl_pnl = np.mean(rl_pnls)
        
        if use_wandb:
            wandb.log({
                "iteration": iteration,
                "avg_rule_pnl": avg_rule_pnl,
                "avg_rl_pnl": avg_rl_pnl,
                "max_rule_pnl": np.max(rule_pnls),
                "max_rl_pnl": np.max(rl_pnls),
                "rl_advantage": avg_rl_pnl - avg_rule_pnl,
                **{f"rule_agent_{i}_pnl": rule_pnls[i] for i in range(n_rule_agents)},
                **{f"rl_agent_{i}_pnl": rl_pnls[i] for i in range(n_rl_agents)}
            })
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}: rule_avg={avg_rule_pnl:.2f}, rl_avg={avg_rl_pnl:.2f}, rl_advantage={avg_rl_pnl-avg_rule_pnl:.2f}")
        
        # save checkpoint (only keep best)
        if avg_rl_pnl > best_rl_pnl:
            best_rl_pnl = avg_rl_pnl
            for f in os.listdir(checkpoint_dir):
                if f.startswith("hybrid_"):
                    os.remove(os.path.join(checkpoint_dir, f))
            
            checkpoint_path = os.path.join(checkpoint_dir, f"hybrid_iter_{iteration}_pnl_{avg_rl_pnl:.2f}.pt")
            torch.save({
                'iteration': iteration,
                'rl_policies': [p.state_dict() for p in rl_policies],
                'rule_params': [a.params for a in rule_agents],
                'avg_rl_pnl': avg_rl_pnl
            }, checkpoint_path)
            print(f"💾 saved checkpoint: {checkpoint_path}")
    
    if use_wandb:
        wandb.finish()
    
    return rl_policies, rule_agents


if __name__ == "__main__":
    print("🔥 training hybrid: rl agents vs evolved strategies...")
    print("this is the ULTIMATE test. may the best algo win.")
    
    train_hybrid(
        n_rule_agents=3,
        n_rl_agents=3,
        n_iterations=50000,  # 50k iterations
        use_wandb=True
    )
    
    print("✅ hybrid training complete!")
