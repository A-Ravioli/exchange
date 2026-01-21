# vectorized exchange using pytorch tensors for 50-100x speedup

from .exchange_vector import VectorizedOrderBook, submit_orders_batch
from .vec_env import VectorizedMultiAgentEnv

# api compatibility - drop-in replacement for the regular version
MultiAgentExchangeEnv = VectorizedMultiAgentEnv
LimitOrderBook = VectorizedOrderBook

__all__ = [
    'VectorizedOrderBook',
    'VectorizedMultiAgentEnv',
    'MultiAgentExchangeEnv',
    'LimitOrderBook',
    'submit_orders_batch',
]
