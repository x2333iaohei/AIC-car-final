"""Causal synchronization and braking metrics derived from recorded telemetry."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from typing import Optional

Second = 1_000_000_000


class StampClock:
    def __init__(self, mode="auto", max_skew=2.):
        self.mode = mode
        self.max_skew = int(max_skew * Second)

    def choose(self, record_ns, header_ns):
        if (self.mode != "bag" and header_ns is not None and header_ns > 0 and
                abs(header_ns - record_ns) <= self.max_skew):
            return header_ns, "header"
        return record_ns, "bag"


@dataclass(frozen=True)
class Sample:
    ns: int
    value: tuple
    clock: str


class Timeline:
    """The telemetry index and ordered scans live on disk. Images are never indexed."""

    def __enter__(self):
        self.temp = TemporaryDirectory(prefix="aic-bag-index-")
        self.db = sqlite3.connect(Path(self.temp.name) / "timeline.sqlite3")
        self.db.execute("CREATE TABLE samples (name TEXT, ns INTEGER, value TEXT, clock TEXT)")
        self.db.execute("CREATE INDEX ordered ON samples (name, ns)")
        return self

    def __exit__(self, *_):
        self.db.close()
        self.temp.cleanup()

    def add(self, name, ns, value, clock):
        self.db.execute("INSERT INTO samples VALUES (?,?,?,?)",
                        (name, ns, json.dumps(value), clock))

    def flush(self):
        self.db.commit()

    @staticmethod
    def _sample(row):
        return Sample(row[0], tuple(json.loads(row[1])), row[2])

    def latest(self, name, ns, age=None):
        clauses = "name=? AND ns<=?"
        args = [name, ns]
        if age is not None:
            clauses += " AND ns>=?"
            args.append(ns - age)
        row = self.db.execute(
            f"SELECT ns,value,clock FROM samples WHERE {clauses} ORDER BY ns DESC,rowid DESC LIMIT 1",
            args).fetchone()
        return self._sample(row) if row else None

    def range(self, name, low=-(2**63), high=2**63 - 1):
        rows = self.db.execute("""SELECT ns,value,clock FROM samples
                                  WHERE name=? AND ns BETWEEN ? AND ? ORDER BY ns,rowid""",
                               (name, low, high))
        for row in rows:
            yield self._sample(row)

    def has(self, name):
        return self.db.execute("SELECT 1 FROM samples WHERE name=? LIMIT 1", (name,)).fetchone() is not None


@dataclass(frozen=True)
class Brake:
    command_ns: Optional[int] = None
    stopped_ns: Optional[int] = None
    duration: Optional[float] = None
    distance: Optional[float] = None
    forward_distance: Optional[float] = None
    status: str = ""


def braking(db, trigger_ns, max_gap=.20, dwell=.20, stop_speed=.02,
            stop_yaw=.05, lookahead=3.):
    """Find a recorded forward-to-zero/reverse transition and integrate to sustained low Odom speed."""
    gap = int(max_gap * Second)
    search_start = trigger_ns - 100_000_000
    previous = db.latest("cmd", search_start - 1, gap)
    command = None
    for sample in db.range("cmd", search_start, trigger_ns + 500_000_000):
        if (previous and previous.value[0] > stop_speed and sample.value[0] <= stop_speed
                and sample.ns - previous.ns <= gap):
            command = sample
            break
        previous = sample
    if command is None:
        return Brake(status="no_confirmed_brake_command")

    start = command.ns
    end = start + int(lookahead * Second)
    resume = next((sample for sample in db.range("cmd", start, end)
                   if sample.ns > start and sample.value[0] > stop_speed), None)
    if resume is not None:
        end = resume.ns - 1
    previous = db.latest("odom", start, gap)
    samples = iter(db.range("odom", start+1, end))
    if previous is None:
        return Brake(start, status="missing_odom_at_command")
    net, forward = 0., 0.
    first_low = None
    for sample in samples:
        delta_ns = sample.ns - previous.ns
        if delta_ns == 0:
            continue
        if delta_ns > gap:
            return Brake(start, status="odom_gap_before_stop")
        v0, v1 = previous.value[0], sample.value[0]
        t0 = previous.ns
        if t0 < start:
            v0 += (v1 - v0) * (start-t0) / delta_ns
            t0 = start
        delta = (sample.ns - t0)/Second
        net += .5*(v0+v1)*delta
        if v0 >= 0. and v1 >= 0.:
            forward += .5*(v0+v1)*delta
        elif max(v0, v1) > 0.:
            # Positive triangular portion of the linearly interpolated interval.
            forward += .5 * max(v0, v1)**2 / abs(v1-v0) * delta

        if abs(v1) <= stop_speed and abs(sample.value[1]) <= stop_yaw:
            if first_low is None:
                first_low = (sample.ns, net, forward)
                if (previous.ns == start and abs(previous.value[0]) <= stop_speed and
                        abs(previous.value[1]) <= stop_yaw):
                    first_low = (start, 0., 0.)
            stop_ns, signed, positive = first_low
            if sample.ns - stop_ns >= int(dwell * Second):
                return Brake(start, stop_ns, (stop_ns-start)/Second, signed, positive,
                             "odom_stop_estimate")
        else:
            first_low = None
        previous = sample
    return Brake(start, status=("command_resumed_before_stop_confirmation" if resume
                                else "no_sustained_stop_in_window"))


_YELLOW_COUNT = re.compile(r"New yellow line detected! Count: (\d+)/(\d+)")
_START_CHANGED = re.compile(r"Start signal changed:\s*(true|false)\s*->\s*(true|false)")


def log_event(text):
    count = _YELLOW_COUNT.search(text)
    if count:
        return "yellow_trigger", int(count[1]), int(count[2]), text
    start_match = _START_CHANGED.search(text)
    if start_match:
        kind = "start_true" if start_match[2] == "true" else "start_false"
        return kind, None, None, text
    for prefix, kind in (
        ("Final yellow line detected.", "yellow_permanent"),
        ("Yellow line stop duration ended.", "yellow_resume"),
        ("Yellow line cleared.", "yellow_clear"),
        ("Yellow reverse brake triggered:", "reverse_trigger"),
        ("Reverse brake ended:", "reverse_end"),
        ("Reverse brake skipped:", "reverse_skipped"),
        ("Previous lap finished.", "lap_reset"),
        ("Line follower is not started", "start_waiting"),
    ):
        if text.startswith(prefix):
            return kind, None, None, text
    return None


class YellowLatch:
    """Image-based candidate events, relative to the beginning of each replay segment."""

    def __init__(self, params):
        self.limit = params["yellow_permanent_count"]
        self.clear_frames = params["yellow_clear_frames"]
        self.duration = params["yellow_stop_duration"]
        self.reset()

    def reset(self):
        self.started = None
        self.latched = self.final = False
        self.clear = self.count = 0
        self.last_trigger = None

    def start(self, value):
        new_lap = self.started is False and value and self.final
        if new_lap:
            self.reset()
        self.started = value
        return new_lap

    def stopping(self, ns):
        return self.final or (self.last_trigger is not None and
                              (ns-self.last_trigger)/Second < self.duration)

    def update(self, ns, detected):
        if self.started is not True:
            return []
        events = []
        if detected:
            self.clear = 0
            if not self.latched:
                self.latched = True
                self.count += 1
                self.final |= self.count >= self.limit
                self.last_trigger = ns
                events.append(("yellow_pixel_trigger", self.count, self.limit))
                if self.final:
                    events.append(("yellow_pixel_permanent", self.count, self.limit))
        elif self.latched:
            self.clear += 1
            if self.clear >= self.clear_frames:
                self.latched = False
                self.clear = 0
                events.append(("yellow_pixel_clear", self.count, self.limit))
        if (not self.final and self.last_trigger is not None and
                (ns - self.last_trigger)/Second >= self.duration):
            events.append(("yellow_pixel_resume", self.count, self.limit))
            self.last_trigger = None
        return events
