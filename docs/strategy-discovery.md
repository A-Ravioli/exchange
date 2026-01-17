# discovering hft strategies with rl and evolution

three approaches to finding profitable trading strategies through competition and self-play.

## the setup

multiple agents compete in the same order book. they see:
- book depth (top 5 levels of bids/asks)
- their own position (inventory, cash)
- other agents' inventories (partial observability)

they choose:
- side (buy or sell)
- price offset from mid
- quantity

rewards are **relative** (zero-sum). your pnl minus the average pnl. forces competition.

## approach 1: evolutionary strategies

evolve rule-based trading strategies using genetic algorithms.

```bash
python src/evolve.py
```

**how it works:**
- start with 50 random agents
- each agent has 12 parameters controlling its behavior
- evaluate fitness by running episodes
- select parents via tournament
- crossover + mutation to create children
- repeat for 100 generations

**strategy encoding:**
agents use simple rules based on:
- order book imbalance (bid pressure vs ask pressure)
- spread size
- current inventory (mean reversion)

parameters control thresholds, sizing, and offsets.

**advantages:**
- fast to evaluate (no backprop)
- interpretable (can inspect parameters)
- good for finding diverse strategies

**output:**
```
Gen 0: best=12.34, mean=-5.67
Gen 10: best=45.23, mean=15.89
...
Gen 50: best=123.45, mean=78.90
```

saves best agent parameters to `best_agent_params.npy`.

## approach 2: reinforcement learning with ppo

train neural network policies using self-play.

```bash
python src/train_rl.py
```

**how it works:**
- 4 agents with separate neural networks
- each network maps observations -> action distribution
- agents compete simultaneously in the same book
- ppo updates with relative rewards
- policies co-evolve through competition

**network architecture:**
- 2-layer mlp (128 hidden units each)
- gaussian policy (continuous actions)
- separate value network for advantage estimation

**self-play dynamics:**
- as agents improve, the environment gets harder
- creates an "arms race" of increasingly sophisticated strategies
- prevents overfitting to static opponents

**advantages:**
- can discover complex strategies
- scales with compute
- handles high-dimensional observations

**output:**
```
Iter 0: PnLs = ['-2.34', '5.67', '-1.23', '4.56']
Iter 10: PnLs = ['12.34', '8.90', '15.67', '10.23']
...
```

saves policies to `policy_agent_0.pt`, etc.

## approach 3: hybrid competition

evolve rule-based agents, then compete them against rl agents.

```bash
python examples/discover_strategies.py
```

**tournament structure:**
1. evolve initial population (30 generations)
2. create variants through mutation
3. compete in round-robin tournament
4. (optional) train rl agents and add to pool
5. select winners, iterate

**cross-pollination:**
- evolved agents provide interpretable baselines
- rl agents can learn to exploit evolved strategies
- creates pressure for both to improve

## what strategies emerge?

some patterns i've observed:

**market making**: post on both sides of the spread, capture rebates. parameters control:
- how far from mid to quote
- quote size
- refresh frequency

**momentum**: detect order flow imbalance, join the dominant side. parameters control:
- imbalance threshold
- position limits
- exit timing

**mean reversion**: inventory management. when long, offer more aggressively. when short, bid more aggressively.

**adversarial**: detect other agents' patterns and front-run them. emerges naturally in self-play.

**spoofing** (kinda): place orders to move the mid, then cancel and trade the other side. harder to do with the current action space but it tries.

## measuring success

beyond raw pnl, look at:
- **sharpe ratio**: pnl / volatility
- **trade count**: how active is the strategy?
- **inventory variance**: is it staying flat or accumulating risk?
- **win rate**: fraction of episodes with positive pnl
- **robustness**: does it work against different opponents?

## scaling up

to discover more sophisticated strategies:

**more agents**: 8-16 agents creates richer dynamics. requires more compute.

**longer episodes**: 1000+ steps lets strategies develop over time.

**diverse opponents**: include market makers, random traders, trend followers.

**meta-learning**: train agents to adapt quickly to new opponents (MAML, reptile).

**reward shaping**: add auxiliary rewards for specific behaviors (providing liquidity, hitting spreads).

**observation augmentation**: add more features (trade history, order book snapshots over time).

**action space expansion**: allow cancellations, multiple simultaneous orders, order modification.

## running experiments

quick test:
```bash
python examples/discover_strategies.py  # ~5 minutes
```

full evolution run:
```bash
python src/evolve.py  # ~30 minutes
```

rl training (longer):
```bash
python src/train_rl.py  # ~2 hours
```

then compete the results:
```python
from evolve import RuleBasedAgent, compete_agents
from train_rl import PolicyNetwork
import torch

# load evolved agent
evolved = RuleBasedAgent(np.load("best_agent_params.npy"))

# load rl policy
rl_policy = PolicyNetwork(obs_dim=26, act_dim=3)
rl_policy.load_state_dict(torch.load("policy_agent_0.pt"))

# compete (need to wrap rl policy in agent class)
# ...
```

## next steps

- implement multi-timeframe observations
- add transaction costs and slippage
- create persistent leaderboard
- visualize strategy behavior
- analyze discovered strategies for insights
- port to rust for 100x speedup

the cool thing about this setup is it's completely self-contained. no external data, no brokerage api, just agents competing in a simulated market. 

and the strategies that emerge are *real*—they work because they exploit actual market microstructure dynamics (order book imbalance, mean reversion, latency arbitrage).

whether they'd work on real exchanges is an empirical question. but at minimum, they're interesting to study.
