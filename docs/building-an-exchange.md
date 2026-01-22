# building an exchange from scratch

or: how i learned to stop worrying and love the order book

## the problem

i wanted to understand how stock exchanges actually work. not the glossy "markets bring buyers and sellers together" explanation you get in econ 101, but the gritty details: what happens when you click "buy" on robinhood? how does the order sit in the book? who matches with who? what if two orders arrive at the exact same time?

reading about it felt insufficient. the only way to really understand something is to build it.[^1]

so i set out to build a toy exchange. not a real one—i have no desire to deal with finra or the sec—but something that captures the essential complexity: a limit order book with realistic matching, different order types, and enough detail that you could plausibly train a trading algorithm on it.

this is the story of how i built it, piece by piece, starting from first principles.

## starting point: what even is an exchange?

at the core, an exchange is stupidly simple. it's just two lists:
- people who want to buy (with their price and quantity)
- people who want to sell (with their price and quantity)

when someone submits an order, you check if it can match with the opposite side. if it can, execute the trade. if it can't, add it to the appropriate list and wait.

that's it. that's the whole thing.

except of course it's not, because the devil is in the details. which brings us to...

## detail #1: the order book structure

first decision: how do we store these lists?

a naive approach would be a simple list of orders, sorted by price. but think about what happens at scale:
- robinhood processes millions of orders per day
- you need to find the best bid/ask instantly
- orders at the same price need to be matched in time order (usually)
- you're constantly adding and removing orders

so we need:
1. fast access to the best bid (highest buy price)
2. fast access to the best ask (lowest sell price)
3. orders at each price level stored in a queue (first in, first out)
4. fast insertion and deletion anywhere in the book

this led me to a sorted dictionary mapping prices to queues:

```python
from sortedcontainers import SortedDict
from collections import deque

bids: SortedDict  # price -> deque of orders
asks: SortedDict  # price -> deque of orders
```

the bids are sorted so the highest price is last (easy to grab). the asks are sorted so the lowest price is first. each price level is a deque—orders join the back, leave from the front. fifo.[^2]

getting the best bid is just `bids.peekitem(-1)[0]`. getting the best ask is `asks.peekitem(0)[0]`. both are O(log n), which is fine.

## detail #2: the order itself

what information does an order need?

```python
@dataclass
class Order:
    id: int                    # unique identifier
    trader_id: int             # who sent this
    side: str                  # "buy" or "sell"
    order_type: str            # more on this later
    price: float               # limit price
    quantity: int              # how many shares
    timestamp: float           # when it arrived
    filled: int = 0            # how much has executed
```

the key insight here is `filled`. orders don't have to execute all at once—they can be partially filled. if i want to buy 100 shares but only 30 are available, i buy those 30 and wait for the rest.

this means every time we match orders, we need to:
1. figure out how much can fill
2. update both orders' `filled` amounts
3. if an order is fully filled, remove it from the book
4. if it's partially filled, leave it there

## detail #3: matching logic

here's where it gets interesting. when a new buy order comes in, we need to check if it can match with existing sell orders.

the basic algorithm:
1. look at the best ask (lowest sell price)
2. if our buy price is >= the ask price, we can match
3. figure out how much quantity can fill (min of what we want and what's available)
4. create a trade
5. update both orders' filled amounts
6. if we still have quantity remaining, repeat with the next price level

```python
def match_order(book, order, current_time):
    trades = []
    contra_side = book.asks if order.side == "buy" else book.bids
    remaining = order.quantity - order.filled
    
    while remaining > 0 and contra_side:
        best_price, queue = contra_side.peekitem(...)  # get best price level
        
        # check if price is acceptable
        if order.side == "buy" and best_price > order.price:
            break  # too expensive
        
        # match against orders in the queue
        for contra_order in queue:
            if remaining == 0:
                break
            
            fill_qty = min(remaining, contra_order.quantity - contra_order.filled)
            # create trade, update fills, emit events...
```

but wait—this is "price-time priority" or fifo matching. the first order at a price level gets filled first. this is how most us equity exchanges work.

some exchanges (especially crypto and derivatives) use "pro rata" matching instead, where fills are allocated proportionally based on order size. so i implemented both:

```python
if book.matching_mode == "fifo":
    return match_fifo(book, order, current_time)
if book.matching_mode == "prorata":
    return match_prorata(book, order, current_time)
```

pro rata is more complex—you have to give the first order in the queue some preferential allocation (usually 40%), then split the rest proportionally among all orders. this prevents people from gaming the system by splitting large orders into tiny pieces.[^3]

## detail #4: order types

"limit order" and "market order" are just the beginning. real exchanges support a menagerie of order types:

**market order**: buy/sell at any price, execute immediately. if you want 100 shares and only 30 exist in the book, you get 30 and the rest is cancelled.

**limit order**: will only execute at your price or better. if it doesn't match immediately, it sits in the book and waits.

**ioc (immediate or cancel)**: execute whatever you can right now, cancel the rest. no resting in the book.

**fok (fill or kill)**: either execute the entire order right now or cancel it. all or nothing.

**post only**: only add to the book, don't match. if this would cross the spread (execute immediately), reject it instead. used by market makers who want to provide liquidity, not take it.

**stop orders**: trigger when the price hits a certain level. stop loss converts to a market order. stop limit converts to a limit order.

**iceberg orders**: show only part of your order size. when that part fills, automatically reveal more from the hidden reserve. prevents front-running of large orders.

each of these required special handling in the matching logic. fok needed a pre-check to see if enough liquidity existed. post only needed a check before matching. stop orders needed a separate queue and a trigger mechanism.

## detail #5: events and market data

an exchange isn't just a matching engine—it's a data firehose. every action generates events:
- order added to book
- order executed (partially or fully)
- order cancelled
- trade occurred
- top of book changed

trading algorithms need this data to make decisions. so i added a callback system:

```python
def emit_market_data(book, event, current_time):
    if book.market_data_callback is None:
        return
    book.market_data_callback(book, event, current_time)
```

simple, but powerful. now algorithms can subscribe to the market data feed and react to changes.

## detail #6: time

here's a subtle but crucial point: exchanges don't just process orders—they process them *in time*.

when you build a real exchange, orders arrive via the network, get queued, and are processed sequentially. there's no actual parallelism.[^4] timestamp order matters enormously.

this means i needed a simulation framework that could:
1. schedule events at specific times
2. process events in chronological order
3. advance time deterministically

enter discrete event simulation:

```python
class Simulation:
    def __init__(self):
        self._queue = []  # min heap of events
        self._time = 0.0
    
    def schedule_event(self, time, callback, *args):
        event = Event(time, seq, callback, args)
        heapq.heappush(self._queue, event)
    
    def run_step(self):
        event = heapq.heappop(self._queue)
        self._time = event.time  # advance time
        event.callback(*event.args)  # fire event
```

now i can schedule things like "submit this order at t=0.5" or "refresh market maker quotes every 0.2 seconds" and everything runs in perfect chronological order.

this is the secret sauce that makes the whole system work. the exchange doesn't run in real time—it runs in simulated time, as fast as your cpu can process events.

## detail #7: trading algorithms

an exchange without traders is just an empty data structure. i needed algorithms that would submit orders and create flow.

**random trader**: submits random orders at regular intervals. picks a random side, random quantity, random price near the mid. simple but effective at creating baseline liquidity.

**market maker**: posts bid and ask quotes on both sides of the spread. refreshes them periodically. tries to capture the spread as profit. this is the bread and butter of how market makers actually operate.

both algorithms track their own state:
- inventory (how many shares they own)
- cash (how much money they've made/lost)
- open orders (so they can cancel them)

and both subscribe to market data events to update their state when trades occur.

## detail #8: observation and interaction

building the exchange was fun, but i wanted to do more with it. specifically, i wanted to train reinforcement learning agents to trade.

this meant wrapping it in a gymnasium environment:

```python
class ExchangeEnv(gym.Env):
    def __init__(self):
        # action: [side, price_offset, quantity]
        self.action_space = spaces.Box(...)
        
        # observation: book depth + position
        self.observation_space = spaces.Box(...)
```

the action space lets an agent choose whether to buy or sell, at what price offset from mid, and how many shares.

the observation space gives the agent:
- top 5 levels of bids (prices and volumes)
- top 5 levels of asks (prices and volumes)
- its current inventory
- its current cash position
- the current mid price

the reward is the pnl change plus a small penalty for holding inventory (encourages the agent to stay relatively flat).

spawn some random traders for liquidity, hook up your favorite rl library, and you're off to the races.

## putting it together

the final architecture looks like this:

```
simulation (discrete event loop)
    ↓
exchange (order book + matching)
    ↓
algorithms (subscribed to market data)
    ↓
more orders → back to exchange
```

everything is deterministic. given the same seed and same order sequence, you get the exact same result every time. this is crucial for debugging and reproducibility.

i can run the whole thing at millions of events per second. way faster than real time. perfect for backtesting or training rl agents.

## things i learned

**1. matching engines are harder than they look**

the basic idea is simple but there are so many edge cases. what if both orders are from the same trader? (self-match prevention) what if an order would cross the spread but is post-only? what if a stop order triggers while processing another order's fills?

each feature interacts with every other feature. the complexity grows quadratically.

**2. discrete event simulation is magic**

once i wrapped my head around it, everything became easier. want to add a new order type? just modify the submit order function. want to add a new algorithm? just schedule its callbacks. want to replay a sequence of events? just feed them to the sim.

separating logical time from wall clock time is incredibly freeing.

**3. market microstructure is fascinating**

building this gave me intuition for how real exchanges work. why do market makers want post-only orders? (so they don't accidentally hit their own quotes when refreshing) why do some exchanges use pro rata matching? (fairer for large orders, harder to game) why do iceberg orders exist? (hide order size from predators)

you can read about this stuff, but building it makes it *click*.

**4. type hints and dataclasses are your friends**

this codebase would be unmaintainable without type hints. when you're juggling orders, trades, events, and callbacks, knowing what type everything is saves your sanity.

dataclasses are perfect for the immutable-ish data structures (orders, trades, events). they generate `__init__`, `__repr__`, and comparison methods automatically. less boilerplate, fewer bugs.

## extensions

things i'd add if i keep working on this:

**auctions**: most exchanges open with an auction to establish a reference price. complex but interesting.

**circuit breakers**: halt trading if price moves too much too fast. requires tracking price history.

**order book visualizations**: animate the book in real time as trades happen. maybe with [manim](https://www.manim.community/)?

**latency simulation**: add random delays to orders based on "distance" from exchange. makes things more realistic.

**multi-asset**: right now it's one order book. real exchanges handle thousands of symbols. would need some refactoring.

**performance optimization**: port the hot paths to rust or c++. could probably get another 10-100x speedup.

## experiments: can we learn to trade?

once i had the exchange working, the obvious next question was: can you train an ai to make money in it?

i ran a bunch of experiments. some worked. some didn't. here's what i learned.

### experiment 1: single agent rl (baseline)

**setup**: one rl agent (ppo with a small 128-unit network) trading against random background traders.

**results**: it learned... something. the agent figured out that inventory penalties are bad, so it mostly stayed flat. occasional small trades. pnl hovered around zero, slightly positive.

**verdict**: boring but stable. the agent basically learned "don't do anything stupid." not terrible, but not interesting.

**problem**: no competition. the random traders are predictable. there's no adversarial pressure pushing the agent to improve.

### experiment 2: multi-agent self-play

**setup**: 4 rl agents competing against each other. same order book, same observations, but they're all trying to maximize their own pnl.

**training details**:
- ppo with clipped advantage
- 3 layers, 128 hidden units
- learning rate 3e-4
- 32 parallel environments (subprocess vectorization)
- 500 steps per episode
- logged to wandb

**results**: way more interesting! agents developed strategies:
- one agent became a passive market maker, posting on both sides
- another became aggressive, taking liquidity
- they learned to react to each other's inventory positions
- average pnl per agent: ~$15-30 per episode (starting from $0)

**verdict**: self-play creates emergent complexity. agents push each other to improve. this is where it got fun.

**insight**: competitive environments force learning. you need adversaries.

### experiment 3: evolution vs reinforcement learning

**setup**: instead of neural networks, evolve simple rule-based strategies using genetic algorithms.

**agent structure**: 12 parameters defining buy/sell thresholds, price offsets, and quantity sizing based on:
- order book imbalance (bid volume vs ask volume)
- spread width
- inventory position (mean reversion)

**evolution details**:
- population size: 50-100
- tournament selection (top 5)
- crossover + mutation (rate: 0.15)
- fitness: pnl + 10×sharpe ratio (rewards consistency)
- 100+ generations

**results**: 
- converged to strategies that lean into imbalance signals
- strong inventory mean reversion (don't accumulate positions)
- preferred tight spreads (more trading opportunities)
- final fitness: ~40-60 (compared to ~0-5 for random)

**verdict**: evolution works! simpler than deep rl, faster to iterate, and surprisingly effective for this problem.

**comparison**: evolved agents matched or beat rl agents with ~1000× fewer parameters. the structure of the rule-based agent was a strong prior.

### experiment 4: scaling up with parallelism

**motivation**: training was slow. 4 agents × 500 steps × single env = not enough data.

**solution**: parallelize everything.

implemented `SubprocVecEnv` (multiprocessing):
- 32 environments running simultaneously
- each with 4 agents
- step all 32 in parallel, collect rollouts, train on combined batch

**speedup**: ~20-30× faster wall clock time (depends on cpu cores)

**training throughput**: 
- before: ~2000 steps/second
- after: ~50,000 steps/second
- could train to convergence in ~30 minutes instead of 10 hours

**other benefits**:
- better gradient estimates (more diverse data)
- smoother learning curves
- less variance in pnl metrics

**lesson**: parallelism isn't just about speed—it's about data quality.

### experiment 5: bigger networks for gpu utilization

**observation**: small 128-unit networks barely use the gpu. most time spent on overhead, not compute.

**hypothesis**: bigger networks would saturate gpu better and learn richer representations.

**architectures tested**:
1. **small**: 2 layers × 128 units (~50k parameters)
2. **large**: 3 layers × [512, 512, 256] (~800k parameters)  
3. **extra large**: 4 layers × [1024, 1024, 512, 256] (~3m parameters)
4. **residual**: 4 residual blocks × 512 units with skip connections

**results**:
- large networks: ~15-20% better final pnl
- extra large: diminishing returns, sometimes overfitting
- residual: similar to large, more stable gradients

**training speed** (with mixed precision on apple mps):
- small: 8k steps/sec
- large: 6k steps/sec  
- xlarge: 3k steps/sec

**verdict**: large networks (512-512-256) hit the sweet spot. better performance without much slowdown.

**why it helps**: agents can memorize opponent behaviors and build better state representations.

### experiment 6: mixed precision training

**idea**: use fp16 instead of fp32 for forward/backward passes to speed up training.

**implementation**: pytorch amp (automatic mixed precision)

```python
with autocast():
    policy_loss = compute_loss(...)
scaler.scale(policy_loss).backward()
scaler.step(optimizer)
```

**results**:
- cuda: ~1.8× speedup 
- apple mps: not supported yet (falls back to fp32)
- cpu: not supported (falls back to fp32)

**stability**: needed gradient clipping (max norm 0.5) to prevent overflow. otherwise fine.

**verdict**: free speedup on cuda. use it if you have nvidia gpus.

### experiment 7: policy diversity and self-play pool

**problem**: agents sometimes collapse to similar strategies. less diversity → less learning pressure.

**solution**: maintain a pool of past best policies. periodically swap opponents.

**implementation**:
- every 50 iterations, clone the best agent
- keep pool of last 10 best policies
- occasionally pit current agents against pool members

**results**: mixed. increased diversity but also increased variance. sometimes helped, sometimes hurt.

**verdict**: interesting idea, needs more tuning. probably works better with longer training runs.

### experiment 8: vectorized order book (gpu acceleration attempt)

**idea**: implement the entire order book in pytorch/cuda for full gpu acceleration.

**implementation**:
- `exchange_vector.py`: order book with tensor operations
- `matching_kernels.py`: custom cuda-like kernels for matching
- batched simulation across multiple environments

**challenges**:
- order books are inherently sequential (time priority matters)
- dynamic data structures (queues of variable length) don't map well to tensors
- lots of branching logic (hard to vectorize)

**results**: 
- worked for simple cases
- ~2-3× slower than cpu version for realistic workloads
- memory hungry (preallocate huge tensors)

**verdict**: gpus aren't magic. some algorithms (like order matching) are fundamentally sequential and don't benefit from parallelism.

**lesson**: profile before optimizing. cpu multiprocessing beat gpu vectorization here.

### experiment 9: hybrid evolutionary rl

**idea**: use evolution to find good parameter initialization, then fine-tune with rl.

**setup**:
1. evolve rule-based agents for 100 generations
2. extract best parameters
3. use them to initialize neural network weights (domain knowledge → network biases)
4. train with ppo

**status**: didn't finish this one. evolutionary strategies were good enough that i got distracted optimizing those instead.

**future work**: probably worth revisiting. combining symbolic (evolution) and neural (rl) representations is powerful.

### what actually worked

**for speed**:
- ✅ parallel environments (massive win)
- ✅ larger networks (modest win)  
- ✅ mixed precision on cuda (good win)
- ❌ gpu vectorization (loss)

**for performance**:
- ✅ multi-agent self-play (essential)
- ✅ evolution for rule-based strategies (surprisingly good)
- ✅ larger networks (better final pnl)
- ⚠️ policy pools (inconsistent)

**for iteration speed**:
- ✅ wandb logging (saved so much debugging time)
- ✅ checkpointing best models
- ✅ discrete event simulation (deterministic replay)

### experiment stats

across all experiments:
- **total training runs**: 30+ (many failed early)
- **total training time**: ~40-50 gpu-hours on m1 mps + some cpu
- **best single agent pnl**: ~$45 per episode (evolved agent)
- **best multi-agent average**: ~$28 per episode (4 rl agents)
- **convergence time**: 200-500 iterations (depends on setup)

### things i'd try next

**better reward shaping**: current reward is just pnl change + inventory penalty. could add:
- sharpe ratio (risk-adjusted returns)
- drawdown penalties
- turnover costs (currently trading is free)

**opponent modeling**: agents could learn to predict other agents' actions and exploit them.

**hierarchical policies**: separate high-level strategy (market making vs taking) from low-level execution (price/quantity).

**imitation learning**: bootstrap from evolved strategies, then improve with rl.

**multi-asset**: trade multiple correlated instruments. way more complex but more realistic.

**latency**: add realistic network delays. changes optimal strategies significantly.

## conclusion

building this took about a week of evenings. the code is ~1000 lines of python across a few files. it's not production-ready—no error handling, no persistence, no gui—but it captures the essential complexity.

most importantly, it's *fun*. there's something deeply satisfying about watching orders flow through the book, trades print, and algorithms make decisions. it's a little economic simulation in a box.

and now when i read about market microstructure or hft strategies or exchange mechanics, i have a mental model to ground it in. i can reason about "what would happen if..." questions by actually trying them in code.

the experiments taught me as much as building the exchange itself. seeing agents learn to trade—or fail to learn—gives you intuition for how markets work. competition creates complexity. simple rules can be surprisingly effective. gpus aren't always the answer.

if you're curious about markets, i'd encourage you to build your own exchange. start simple—just bids, asks, and basic matching. add features as you get curious about them. then try to train something to trade in it. you'll learn way more than you would from reading textbooks.

the code is on [github](https://github.com) if you want to poke around.[^5]

---

[^1]: feynman supposedly said "what i cannot create, i do not understand." this is probably apocryphal but it's a good heuristic anyway.

[^2]: some exchanges use price-time-size priority where larger orders get preference. others use pure pro rata. others use weird hybrid schemes. markets are surprisingly diverse.

[^3]: in the early days of electronic trading, people would split a 10,000 share order into 10,000 one-share orders to get more queue position under pro rata. exchanges got wise to this and added minimum allocation amounts.

[^4]: some modern exchanges have parallel matching engines for different symbols, but within a single symbol, everything is sequential. you can't have race conditions in the order book.

[^5]: i also implemented this in rust and julia. the rust version is about 30x faster. highly recommend if you want to run millions of simulations.
