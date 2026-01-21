"""
Enhanced network architectures for better GPU utilization.
Larger networks with more parameters to saturate GPU compute.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class LargePolicyNetwork(nn.Module):
    """
    Large policy network with 3 hidden layers (512, 512, 256).
    Better GPU utilization than small 128-unit networks.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=[512, 512, 256]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)
    
    def act(self, obs, deterministic=False):
        """Sample action from policy. Compatible with PolicyNetwork API."""
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class LargeValueNetwork(nn.Module):
    """
    Large value network matching policy network capacity.
    """
    
    def __init__(self, obs_dim: int, hidden_dims=[512, 512, 256]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ExtraLargePolicyNetwork(nn.Module):
    """
    Extra large policy network for maximum GPU utilization.
    4 hidden layers (1024, 1024, 512, 256).
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=[1024, 1024, 512, 256]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # Add layer norm for stability
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(in_dim, act_dim)
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


class ExtraLargeValueNetwork(nn.Module):
    """Extra large value network."""
    
    def __init__(self, obs_dim: int, hidden_dims=[1024, 1024, 512, 256]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ResidualPolicyNetwork(nn.Module):
    """
    Policy network with residual connections for deeper architectures.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim=512, n_blocks=4):
        super().__init__()
        
        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs):
        h = torch.relu(self.input_layer(obs))
        
        for block in self.blocks:
            h = block(h)
        
        mean = self.mean(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)
    
    def act(self, obs, deterministic=False):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return torch.relu(out + residual)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_networks():
    """Test network architectures and print parameter counts."""
    from train_rl import PolicyNetwork as SmallPolicyNetwork, ValueNetwork as SmallValueNetwork
    
    obs_dim = 24
    act_dim = 3
    
    print("Network Architecture Comparison")
    print("=" * 60)
    
    # Small networks (current)
    small_policy = SmallPolicyNetwork(obs_dim, act_dim, hidden=128)
    small_value = SmallValueNetwork(obs_dim, hidden=128)
    print(f"Small Policy:  {count_parameters(small_policy):>10,} parameters")
    print(f"Small Value:   {count_parameters(small_value):>10,} parameters")
    print(f"Total Small:   {count_parameters(small_policy) + count_parameters(small_value):>10,} parameters")
    
    print()
    
    # Large networks
    large_policy = LargePolicyNetwork(obs_dim, act_dim)
    large_value = LargeValueNetwork(obs_dim)
    print(f"Large Policy:  {count_parameters(large_policy):>10,} parameters")
    print(f"Large Value:   {count_parameters(large_value):>10,} parameters")
    print(f"Total Large:   {count_parameters(large_policy) + count_parameters(large_value):>10,} parameters")
    
    print()
    
    # Extra large networks
    xlarge_policy = ExtraLargePolicyNetwork(obs_dim, act_dim)
    xlarge_value = ExtraLargeValueNetwork(obs_dim)
    print(f"XLarge Policy: {count_parameters(xlarge_policy):>10,} parameters")
    print(f"XLarge Value:  {count_parameters(xlarge_value):>10,} parameters")
    print(f"Total XLarge:  {count_parameters(xlarge_policy) + count_parameters(xlarge_value):>10,} parameters")
    
    print()
    
    # Residual networks
    res_policy = ResidualPolicyNetwork(obs_dim, act_dim)
    print(f"Residual Policy: {count_parameters(res_policy):>10,} parameters")
    
    print("=" * 60)
    
    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, obs_dim)
    
    # Test large policy
    action, log_prob = large_policy.act(obs)
    assert action.shape == (batch_size, act_dim), f"Wrong action shape: {action.shape}"
    assert log_prob.shape == (batch_size,), f"Wrong log_prob shape: {log_prob.shape}"
    
    # Test large value
    value = large_value(obs)
    assert value.shape == (batch_size,), f"Wrong value shape: {value.shape}"
    
    print("✅ All network architectures working correctly!")


if __name__ == "__main__":
    test_networks()
