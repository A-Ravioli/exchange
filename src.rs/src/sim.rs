use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub struct Event {
    pub time: f64,
    pub seq: u64,
    pub callback: Box<dyn FnMut()>,
}

impl Eq for Event {}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .total_cmp(&self.time)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Simulation {
    queue: BinaryHeap<Event>,
    seq: u64,
    time: f64,
    end_time: f64,
}

impl Simulation {
    pub fn new(end_time: f64) -> Self {
        Self {
            queue: BinaryHeap::new(),
            seq: 0,
            time: 0.0,
            end_time,
        }
    }

    pub fn current_time(&self) -> f64 {
        self.time
    }

    pub fn schedule_event<F>(&mut self, time: f64, callback: F)
    where
        F: FnMut() + 'static,
    {
        if time < self.time {
            return;
        }
        let event = Event {
            time,
            seq: self.seq,
            callback: Box::new(callback),
        };
        self.seq += 1;
        self.queue.push(event);
    }

    pub fn run_step(&mut self) -> Option<f64> {
        if let Some(mut event) = self.queue.pop() {
            if event.time > self.end_time {
                self.queue.push(event);
                return None;
            }
            self.time = event.time;
            (event.callback)();
            return Some(self.time);
        }
        None
    }

    pub fn run_until(&mut self, end_time: f64) {
        self.end_time = end_time;
        while !self.queue.is_empty() {
            if self.run_step().is_none() {
                break;
            }
        }
    }

    pub fn is_done(&self) -> bool {
        self.queue.is_empty() || self.time >= self.end_time
    }
}

pub fn init_sim(end_time: f64) -> Simulation {
    Simulation::new(end_time)
}
