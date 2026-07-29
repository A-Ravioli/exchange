# exchange

creé clones del NYSE y el CME, originalmente para un proyecto de investigación con los genios del laboratorio de Teoría de la Información Cuántica de UMass Amherst, y ahora lo estoy liberando como código abierto.

construí la versión original en julia bastante rápido durante un fin de semana, pero julia es un poco mala como lenguaje, así que lo estamos reconstruyendo en python para que más personas puedan usarlo y para que yo pueda envolverlo en un entorno de gymnasium para algunos experimentos geniales de descubrimiento de estrategias naturales de HFT/MFT mediante rl/ml, ¡los cuales muestro en la sección de experimentos!

escribí una breve publicación en el blog sobre cómo funcionan los exchanges y te guía sobre cómo derivar cada característica lógicamente (desde los primeros principios y esas cosas). espero que ese blogpost esté enlazado aquí en algún lugar, de alguna manera, eventualmente.

# repo structure

# cómo orientarse en el repo

aquí tienes un pequeño árbol relajado del repo.

```bash
src/                    # implementación central en python
    exchange.py         # libro de órdenes y motor de emparejamiento (fifo + prorata)
    sim.py              # simulación de eventos discretos
    algorithms.py       # algoritmos de trading (market maker, random trader)
    visualizer.py       # visualización del libro en la terminal
    gym_env.py          # wrapper de gymnasium para agente único
    multi_agent_env.py  # entorno competitivo multi-agente
    parallel_env.py     # wrapper de entorno paralelo para entrenamiento más rápido
    networks.py         # arquitecturas de redes neuronales para rl
    evolve.py           # estrategias evolutivas para descubrir reglas de trading
    test_*.py           # pruebas

src/vector/             # entorno de entrenamiento con tensores por lotes (batched)
    env.py              # simulador de rl multi-agente vectorizado simplificado

src.jl/                 # versión original en julia

src.rs/                 # versión acelerada en rust

train.py                # script de entrenamiento unificado (rl, evolución, híbrido)

examples/               # ejemplos de uso
    gym_example.py           # ejemplo básico de uso de rl
    discover_strategies.py   # pipeline completo de descubrimiento de estrategias
    test_multi_agent.py      # prueba de competición multi-agente

docs/                   # documentación
    building-an-exchange.md  # guía paso a paso de cómo se construyó esto
```

# quick start

ejecuta una simulación básica:
```python
# cd into src
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
```

# making this a gym environment

envolví el exchange en un entorno de gymnasium para que puedas entrenar agentes de rl en él. versión de agente único:

```python
from src.gym_env import ExchangeEnv

env = ExchangeEnv(max_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # [side, price_offset, quantity]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.render()  # muestra el libro de órdenes + tu posición
```

# training agents

usa el script de entrenamiento consolidado para todos los modos:

```bash
# entrenamiento de rl con entornos de multiprocesamiento (por defecto)
python3 train.py --mode rl --n_agents 4 --n_iterations 1000 --n_envs 32

# entrenamiento de rl con el simulador de entrenamiento vectorizado simplificado
python3 train.py --mode rl --env vector --device auto --n_agents 4 --n_iterations 1000 --n_envs 64

# redes de política/valor de red de Kolmogorov-Arnold vía PyKAN
python3 train.py --mode rl --env vector --network_size kan --n_envs 64 --n_iterations 20 --steps_per_iter 32

# estrategias evolutivas
python3 train.py --mode evolution --n_agents 8 --n_iterations 500

# modo híbrido (rl + evolución)
python3 train.py --mode hybrid --n_agents 4 --n_iterations 1000

# opciones
python3 train.py --help
```

el script de entrenamiento incluye:
- entornos paralelos para una recolección de datos más rápida
- redes neuronales más grandes para mayor capacidad
- Redes de Kolmogorov-Arnold opcionales a través de PyKAN
- entrenamiento de precisión mixta (solo cuda)
- ppo de mini-lote con múltiples épocas
- integración con wandb para el registro (logging)
- backend de entrenamiento vectorizado opcional optimizado para apple mps/cuda/cpu

## ¿qué estrategias emergen?

al ejecutar estos experimentos, he visto:
- **market making**: publicar en ambos lados, capturar el spread
- **momentum**: detectar el desequilibrio del flujo de órdenes y unirse a él
- **reversión a la media**: gestionar el inventario operando en sentido contrario a tu posición
- **adversarial**: aprender a explotar los patrones de otros agentes
- **quasi-spoofing**: colocar órdenes para mover el mid, luego operar el otro lado (más o menos)

el entorno multi-agente obliga a los agentes a competir, por lo que desarrollan estrategias que realmente funcionan contra oponentes inteligentes, no solo contra ruido aleatorio.

# features

**capacidades del exchange:**
- emparejamiento por prioridad de precio-tiempo (fifo)
- emparejamiento pro-rata con asignación de primera orden
- prevención de auto-emparejamiento (self-match prevention)
- órdenes market, limit, ioc, fok, post-only
- órdenes stop loss y stop limit
- órdenes iceberg con cantidad oculta

**simulación:**
- programación de eventos discretos (min-heap)
- avance de tiempo determinista
- se ejecuta mucho más rápido que el tiempo real (~500k eventos/seg en python)

**implementación vectorizada:**
- simulador de entrenamiento de rl simplificado utilizando tensores de pytorch por lotes
- un libro de tensores a través de muchos entornos
- liquidez de fondo y flujo de mercado para señales de aprendizaje densas
- benchmark con `python3 scripts/benchmark_envs.py`

**descubrimiento de estrategias:**
- estrategias evolutivas (algoritmos genéticos)
- self-play ppo (redes neuronales)
- entornos competitivos multi-agente
- evaluación por torneo

# docs

- [building an exchange](docs/building-an-exchange.md) - guía paso a paso de cómo construí esto desde los primeros principios

# dependencies

```bash
pip install numpy sortedcontainers gymnasium torch wandb cloudpickle
```

# performance

- python: ~500k eventos/seg (m1 mac)
- vectorizado: benchmark localmente con `python3 scripts/benchmark_envs.py`

# license
está bajo licencia MIT. no hagas nada raro. la licencia está en [LICENSE](LICENSE)
