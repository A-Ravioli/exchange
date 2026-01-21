"""
Parallel environment wrapper for running multiple environments simultaneously.
Implements SubprocVecEnv pattern for CPU parallelism.
"""

from __future__ import annotations

import numpy as np
import multiprocessing as mp
from typing import Callable, List, Dict, Any, Optional
import cloudpickle


def worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, 
           env_fn: Callable):
    """Worker process for running a single environment."""
    parent_remote.close()
    env = env_fn()
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, rewards, dones, truncs, infos = env.step(data)
                remote.send((obs, rewards, dones, truncs, infos))
                
            elif cmd == 'reset':
                obs, info = env.reset(seed=data)
                remote.send((obs, info))
                
            elif cmd == 'close':
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        print(f"Worker got KeyboardInterrupt")
    finally:
        env.close() if hasattr(env, 'close') else None


class SubprocVecEnv:
    """
    Vectorized environment that runs multiple environments in parallel subprocesses.
    
    Args:
        env_fns: List of functions that create environments
        context: Multiprocessing context ('spawn' or 'fork')
    """
    
    def __init__(self, env_fns: List[Callable], context='spawn'):
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)
        
        # Use spawn context for compatibility (works on all platforms)
        ctx = mp.get_context(context)
        
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, cloudpickle.dumps(env_fn))
            process = ctx.Process(target=_worker_shared_memory, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get spaces from first environment
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        
    def reset(self, seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Reset all environments and return batched observations."""
        seeds = [seed + i if seed is not None else None for i in range(self.n_envs)]
        
        for remote, s in zip(self.remotes, seeds):
            remote.send(('reset', s))
        
        results = [remote.recv() for remote in self.remotes]
        obs_list = [result[0] for result in results]
        
        # Batch observations from all environments
        return self._batch_observations(obs_list)
    
    def step(self, actions: List[Dict[int, np.ndarray]]) -> tuple:
        """
        Step all environments with given actions.
        
        Args:
            actions: List of action dicts (one per env)
            
        Returns:
            Batched observations, rewards, dones, truncs, infos
        """
        # Send actions to all workers
        for i, remote in enumerate(self.remotes):
            remote.send(('step', actions[i]))
        
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs_list, rewards_list, dones_list, truncs_list, infos_list = zip(*results)
        
        return (
            self._batch_observations(obs_list),
            self._batch_rewards(rewards_list),
            self._batch_dones(dones_list),
            self._batch_dones(truncs_list),
            list(infos_list)
        )
    
    def _batch_observations(self, obs_list: List[Dict]) -> Dict[int, np.ndarray]:
        """Batch observations from multiple environments."""
        if not obs_list:
            return {}
        
        # Assume all envs have same agents
        first_obs = obs_list[0]
        n_agents = len(first_obs)
        
        batched = {}
        for agent_id in first_obs.keys():
            # Stack observations for this agent across all envs
            agent_obs = np.stack([obs[agent_id] for obs in obs_list], axis=0)
            batched[agent_id] = agent_obs
        
        return batched
    
    def _batch_rewards(self, rewards_list: List[Dict]) -> Dict[int, np.ndarray]:
        """Batch rewards from multiple environments."""
        if not rewards_list:
            return {}
        
        first_rewards = rewards_list[0]
        batched = {}
        
        for agent_id in first_rewards.keys():
            agent_rewards = np.array([rewards[agent_id] for rewards in rewards_list])
            batched[agent_id] = agent_rewards
        
        return batched
    
    def _batch_dones(self, dones_list: List[Dict]) -> Dict[int, np.ndarray]:
        """Batch done flags from multiple environments."""
        if not dones_list:
            return {}
        
        first_dones = dones_list[0]
        batched = {}
        
        for agent_id in first_dones.keys():
            agent_dones = np.array([dones[agent_id] for dones in dones_list])
            batched[agent_id] = agent_dones
        
        return batched
    
    def close(self):
        """Close all worker processes."""
        if self.closed:
            return
        
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()
        
        self.closed = True
    
    def __del__(self):
        if not self.closed:
            self.close()


def _worker_shared_memory(remote, parent_remote, env_fn_wrapper):
    """Worker with pickled environment function."""
    parent_remote.close()
    env_fn = cloudpickle.loads(env_fn_wrapper)
    env = env_fn()
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                result = env.step(data)
                remote.send(result)
                
            elif cmd == 'reset':
                result = env.reset(seed=data)
                remote.send(result)
                
            elif cmd == 'close':
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close() if hasattr(env, 'close') else None


class ParallelEnv:
    """
    High-level wrapper for parallel environments with simplified API.
    Automatically handles batching and unbatching.
    """
    
    def __init__(self, env_fns: List[Callable], n_envs: int = 32):
        """
        Args:
            env_fns: List of environment creation functions
            n_envs: Number of parallel environments
        """
        if len(env_fns) != n_envs:
            # Replicate the first function if only one provided
            if len(env_fns) == 1:
                env_fns = env_fns * n_envs
            else:
                raise ValueError(f"Expected {n_envs} env functions, got {len(env_fns)}")
        
        self.n_envs = n_envs
        self.vec_env = SubprocVecEnv(env_fns)
        self.observation_space = self.vec_env.observation_space
        self.action_space = self.vec_env.action_space
    
    def reset(self, seed: Optional[int] = None):
        """Reset all environments."""
        return self.vec_env.reset(seed=seed)
    
    def step(self, actions):
        """
        Step all environments.
        
        Args:
            actions: Dict mapping agent_id to action (same for all envs)
                     OR list of dicts (one per env)
        """
        # Convert to list of action dicts (one per env)
        if isinstance(actions, list):
            action_list = actions
        else:
            # Same action dict for all envs
            action_list = [actions for _ in range(self.n_envs)]
        
        return self.vec_env.step(action_list)
    
    def close(self):
        """Close all environments."""
        self.vec_env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
