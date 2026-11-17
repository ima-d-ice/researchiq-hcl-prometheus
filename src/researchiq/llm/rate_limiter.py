"""Rolling 60-second token + request limiter shared by every Groq call in this process."""
import logging
import threading
import time
from collections import deque
from typing import Callable

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, tpm: int, rpm: int, tpd: int, margin: float = 0.9,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.tpm_cap = int(tpm * margin)
        self.rpm_cap = max(1, int(rpm * margin))
        self.tpd = tpd
        self._clock, self._sleep = clock, sleep
        self._events: deque[list] = deque()  # [timestamp, tokens]
        self._lock = threading.Lock()
        self.run_tokens = 0

    def _evict(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60:
            self._events.popleft()

    def used(self) -> int:
        self._evict(self._clock())
        return sum(e[1] for e in self._events)

    def try_acquire(self, estimate: int) -> tuple[list | None, float]:
        """Non-blocking. Returns (handle, 0) if it fits now, else (None, seconds_to_wait)."""
        if estimate > self.tpm_cap:
            raise ValueError(f"request of ~{estimate} tokens can never fit in {self.tpm_cap} TPM")
        with self._lock:
            now = self._clock()
            self._evict(now)
            used = sum(e[1] for e in self._events)
            if used + estimate <= self.tpm_cap and len(self._events) < self.rpm_cap:
                ev = [now, estimate]
                self._events.append(ev)
                return ev, 0.0
            return None, (60 - (now - self._events[0][0])) if self._events else 0.1

    def acquire(self, estimate: int) -> list:
        """Block until `estimate` tokens fit in the rolling window; returns a handle."""
        while True:
            handle, wait = self.try_acquire(estimate)
            if handle is not None:
                return handle
            log.info("Groq budget: %d/%d tokens used in last 60s, waiting %.1fs",
                     self.used(), self.tpm_cap, wait)
            self._sleep(max(wait, 0.1))

    def settle(self, handle: list, actual: int | None) -> None:
        """Replace the estimate with the real usage once the call returns."""
        with self._lock:
            if actual is not None:
                handle[1] = actual
            self.run_tokens += handle[1]
            if self.run_tokens > self.tpd:
                log.warning("Groq daily token budget (%d) exceeded this run", self.tpd)


class KeyPool:
    """Several API keys, each with its own rolling limiter. A key that hits 429 cools down
    and the call moves to the next key."""

    def __init__(self, keys: list[str], tpm: int, rpm: int, tpd: int,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.keys = keys or [""]
        self.limiters = [RateLimiter(tpm, rpm, tpd, clock=clock, sleep=sleep) for _ in self.keys]
        self._cool = [0.0] * len(self.keys)
        self._clock, self._sleep = clock, sleep
        self._next = 0
        self._lock = threading.Lock()
        self.tpm_cap = self.limiters[0].tpm_cap
        self.run_tokens = 0

    def acquire(self, estimate: int) -> tuple[int, list]:
        """Returns (key_index, handle). Blocks only if every key is busy or cooling down."""
        while True:
            now = self._clock()
            waits = []
            with self._lock:
                start = self._next
            for off in range(len(self.keys)):
                i = (start + off) % len(self.keys)
                if self._cool[i] > now:
                    waits.append(self._cool[i] - now)
                    continue
                handle, wait = self.limiters[i].try_acquire(estimate)
                if handle is not None:
                    with self._lock:
                        self._next = (i + 1) % len(self.keys)  # spread load round-robin
                    return i, handle
                waits.append(wait)
            w = max(min(waits), 0.1)
            log.info("All %d Groq key(s) busy, waiting %.1fs", len(self.keys), w)
            self._sleep(w)

    def settle(self, idx: int, handle: list, actual: int | None) -> None:
        before = self.limiters[idx].run_tokens
        self.limiters[idx].settle(handle, actual)
        self.run_tokens += self.limiters[idx].run_tokens - before

    def cooldown(self, idx: int, seconds: float) -> None:
        self._cool[idx] = self._clock() + seconds
        log.warning("Groq key #%d rate limited; rotating (cooldown %.0fs)", idx + 1, seconds)
