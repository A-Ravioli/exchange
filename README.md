# exchange

i made clones of the NYSE and CME, originally for a research project with the very fun people at UMass Amherst's Quantum Information Theory lab, and am now open-sourcing it.

i built the original version in julia kinda quickly over a weekend, but julia kinda sucks as a language so we're rebuilding it in python so that more people can use it and so that i can wrap it in a gymnasium environment for some cool rl/ml natural HFT/MFT strategy discovery experiments which i show off in the experiments section!

i wrote a short blog post on how the exchanges work and it walks you through how to derive each feature logically (first principles-y and stuff). that blogpost will be linked here somewhere somehow eventually i hope. 

# repo structure

# how to make your way around the repo

here's a chill lil tree for the repo.

```bash
src/                    # core python implementation
    exchange.py         # order book and matching engine (fifo + prorata)
    sim.py              # discrete event simulation
    algorithms.py       # trading algorithms (market maker, random trader)
    visualizer.py       # terminal visualization of the book
    gym_env.py          # single-agent gymnasium wrapper
    multi_agent_env.py  # multi-agent competitive environment
    parallel_env.py     # parallel environment wrapper for faster training
    networks.py         # neural network architectures for rl
    evolve.py           # evolutionary strategies for discovering trading rules
    test_*.py           # tests

src.vector/             # gpu-accelerated vectorized implementation
    exchange_vector.py  # vectorized order book using pytorch tensors
    vec_env.py          # vectorized multi-agent environment
    batch_sim.py        # batched simulation for parallel training
    matching_kernels.py # optimized matching algorithms

src.jl/                 # original version in julia

src.rs/                 # accelerated version in rust

train.py                # unified training script (rl, evolution, hybrid)

examples/               # usage examples
    gym_example.py           # basic rl usage example
    discover_strategies.py   # full strategy discovery pipeline
    test_multi_agent.py      # test multi-agent competition

docs/                   # documentation
    building-an-exchange.md  # walkthrough of how this was built
    strategy-discovery.md    # guide to rl and evolution experiments
```

# quick start

run a basic simulation:
```bash
cd src
python -c "
from exchange import init_exchange
from sim import init_sim
from algorithms import RandomTrader, MarketMaker

book = init_exchange(tick_size=0.01)
sim = init_sim(end_time=100.0)

# add some traders
RandomTrader(0, book, sim, interval=0.5)
MarketMaker(1, book, sim, spread=0.1)

sim.run_until(100.0)

# visualize the book
from visualizer import visualize_book
visualize_book(book)
"
```

# making this a gym environment

wrapped the exchange in a gymnasium environment so you can train rl agents on it. single-agent version:

```python
from src.gym_env import ExchangeEnv

env = ExchangeEnv(max_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # [side, price_offset, quantity]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.render()  # shows the order book + your position
```

# training agents

use the consolidated training script for all modes:

```bash
# rl training with all optimizations (default)
python train.py --mode rl --n_agents 4 --n_iterations 1000 --n_envs 32

# evolutionary strategies
python train.py --mode evolution --n_agents 8 --n_iterations 500

# hybrid mode (rl + evolution)
python train.py --mode hybrid --n_agents 4 --n_iterations 1000

# options
python train.py --help
```

the training script includes:
- parallel environments for faster data collection
- larger neural networks for better capacity
- mixed precision training (cuda only)
- mini-batch ppo with multiple epochs
- wandb integration for logging

## what strategies emerge?

from running these experiments, i've seen:
- **market making**: post on both sides, capture the spread
- **momentum**: detect order flow imbalance and join it
- **mean reversion**: manage inventory by trading opposite your position
- **adversarial**: learn to exploit other agents' patterns
- **quasi-spoofing**: place orders to move the mid, then trade the other side (kinda)

the multi-agent environment forces agents to compete, so they evolve strategies that actually work against intelligent opponents, not just random noise.

# features

**exchange capabilities:**
- price-time priority (fifo) matching
- pro-rata matching with top-order allocation
- self-match prevention
- market, limit, ioc, fok, post-only orders
- stop loss and stop limit orders
- iceberg orders with hidden quantity

**simulation:**
- discrete event scheduling (min-heap)
- deterministic time advancement
- runs way faster than real time (~500k events/sec in python)

**vectorized implementation:**
- gpu-accelerated order book using pytorch tensors
- 50-100x speedup over regular python
- batch processing of multiple orders
- parallel environment execution

**strategy discovery:**
- evolutionary strategies (genetic algorithms)
- ppo self-play (neural networks)
- multi-agent competitive environments
- tournament evaluation

# docs

- [building an exchange](docs/building-an-exchange.md) - walks through how i built this from first principles
- [strategy discovery](docs/strategy-discovery.md) - guide to the rl and evolution experiments

# dependencies

```bash
pip install numpy sortedcontainers gymnasium torch wandb
```

# performance

- python: ~500k events/sec (m1 mac)
- vectorized (gpu): 50-100x faster with batching

# license
it's MIT licensed. don't do anything weird. the license is in [LICENSE](LICENSE)
