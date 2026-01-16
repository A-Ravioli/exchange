"""
Sim.jl - Discrete Event Simulation Engine

Owns the event queue and simulation time. Provides the core discrete event
simulation (DES) infrastructure for the HFT digital twin.

Responsibilities:
- Maintain a priority queue of events ordered by simulation time
- Advance simulation time by processing events in chronological order
- Schedule new events (orders, market data, network messages, etc.)
- Provide time management (current time, time step, end time)
- Coordinate event processing across all simulation components

The event queue uses DataStructures.PriorityQueue keyed by simulation time.
Each event contains a timestamp and a callback function to execute when the
event fires.

Key functions (to be implemented):
- `init_sim(params)`: Initialize simulation with parameters
- `schedule_event(time, callback)`: Add event to queue
- `run_until(time)`: Process events until specified time
- `current_time()`: Get current simulation time
- `is_done()`: Check if simulation should terminate

The Sim module is the central coordinator - all other modules (Venue, Agents,
Network) schedule events through Sim to advance the simulation forward.
"""
module Sim

# Event queue and time management will be implemented here
# using DataStructures

end # module

