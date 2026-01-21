"""
Vectorized exchange implementation using PyTorch tensors.
GPU-accelerated order book and matching for 50-100x speedup.
"""

from .exchange_vector import VectorizedOrderBook, submit_orders_batch
from .vec_env import VectorizedMultiAgentEnv

# API compatibility - can be used as drop-in replacement
MultiAgentExchangeEnv = VectorizedMultiAgentEnv
LimitOrderBook = VectorizedOrderBook

__all__ = [
    'VectorizedOrderBook',
    'VectorizedMultiAgentEnv',
    'MultiAgentExchangeEnv',
    'LimitOrderBook',
    'submit_orders_batch',
]
