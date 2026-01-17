# exchange

i made clones of the NYSE and CME, originally for a research project with the very fun people at UMass Amherst's Quantum Information Theory lab, and am now open-sourcing it.

i built the original version in julia kinda quickly over a weekend, but julia kinda sucks as a language so we're rebuilding it in python so that more people can use it and so that i can wrap it in a gymnasium environment for some cool rl/ml natural HFT/MFT strategy discovery experiments which i show off in the experiments section!

i wrote a short blog post on how the exchanges work and it walks you through how to derive each feature logically (first principles-y and stuff). that blogpost will be linked here somewhere somehow eventually i hope. 

i will later finish rebuilding this in rust so that i can speed up the execution speed of this significantly to scale RL faster.

# how to make your way around the repo

here's a chill lil tree for the repo.

```bash
src/ // core python implementation
    exchange.py        // order book and matching engine (fifo + prorata)
    sim.py             // discrete event simulation
    algorithms.py      // trading algorithms (market maker, random trader)
    visualizer.py      // terminal visualization of the book
    gym_env.py         // single-agent gymnasium wrapper
    multi_agent_env.py // multi-agent competitive environment
    train_rl.py        // ppo self-play training
    evolve.py          // evolutionary strategies for discovering trading rules

examples/
    gym_example.py           // basic rl usage example
    discover_strategies.py   // full strategy discovery pipeline
    test_multi_agent.py      // test multi-agent competition

docs/
    building-an-exchange.md  // walkthrough of how this was built (that blog post!)
    strategy-discovery.md    // guide to rl and evolution experiments

src.jl/ // in julia
    Exchange.jl     // order book, but in julia
    Sim.jl          // discrete event sim
    Algorithms.jl   // trading strategies
    Visualizer.jl   // plotting

src.rs/ // in rust, so faster, i may delete this or have the python version wrap it
    exchange.rs // order book, but 30x faster
    sim.rs      // discrete event sim
    py.rs       // python bindings via pyo3
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
from gym_env import ExchangeEnv

env = ExchangeEnv(max_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # [side, price_offset, quantity]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.render()  # shows the order book + your position
```

# discovering hft strategies with rl and evolution

here's where it gets fun. i built three ways to discover trading strategies through competition:

## 1. evolutionary strategies

genetic algorithms for rule-based strategies. fast and interpretable.

```bash
python src/evolve.py
```

evolves 50 agents over 100 generations. each agent has 12 parameters controlling:
- when to buy vs sell based on order book imbalance
- how far from mid to place orders
- position sizing
- inventory mean reversion

saves the best agent's parameters to `best_agent_params.npy`.

## 2. rl with self-play

train neural network policies that compete against each other.

```bash
python src/train_rl.py
```

4 agents with separate mlp policies compete in the same order book. uses ppo with relative rewards (zero-sum). as agents get better, the environment gets harder. classic self-play dynamics.

saves trained policies to `policy_agent_0.pt`, etc.

## 3. hybrid competition

combine both approaches:

```bash
python examples/discover_strategies.py
```

evolves initial strategies, creates variants through mutation, then runs tournaments to find winners.

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
uv venv
uv pip install numpy sortedcontainers gymnasium torch
```

# performance

- python: ~500k events/sec (m1 mac)
- rust: ~15m events/sec (30x faster)
- julia: ~8m events/sec (16x faster) 

# license
it's MIT licensed. don't do anything weird. the license is in [LICENSE](LICENSE)
