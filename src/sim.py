from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import heapq
import math

# discrete event simulation - runs callbacks at scheduled times

@dataclass(order=True)
class Event:
    time: float  # when this event fires
    seq: int  # breaks ties so events fire in order added
    callback: Callable[..., Any]  # what to run
    args: Tuple[Any, ...]
    kwargs: dict


class Simulation:
    def __init__(self, end_time: float = math.inf) -> None:
        self._queue: List[Event] = []  # min heap of events
        self._seq = 0  # counter for event ordering
        self._time = 0.0  # current sim time
        self._end_time = end_time

    @property
    def current_time(self) -> float:
        return self._time

    def schedule_event(self, time: float, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        # add an event to the queue
        if time < self._time:
            raise ValueError("Cannot schedule event in the past")
        event = Event(time, self._seq, callback, args, kwargs)
        self._seq += 1
        heapq.heappush(self._queue, event)

    def run_step(self) -> Optional[Event]:
        # run the next event in the queue
        if not self._queue:
            return None
        event = heapq.heappop(self._queue)
        if event.time > self._end_time:
            heapq.heappush(self._queue, event)  # put it back
            return None
        self._time = event.time  # advance sim time
        event.callback(*event.args, **event.kwargs)  # fire the event
        return event

    def run_until(self, end_time: Optional[float] = None) -> None:
        # run simulation until end time or queue empty
        if end_time is not None:
            self._end_time = end_time
        while self._queue and self._time <= self._end_time:
            if self.run_step() is None:
                break

    def is_done(self) -> bool:
        return not self._queue or self._time >= self._end_time


def init_sim(end_time: float = math.inf) -> Simulation:
    return Simulation(end_time=end_time)
