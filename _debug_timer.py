from typing import Optional, Dict, NamedTuple
from time import time as time_now


class StoppingTimerBeforeStartingException(Exception):
    pass


class SingleTimer:

    def __init__(self, name: str):
        self._name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @property
    def elapsed_time(self) -> Optional[float]:
        if (self._start_time is not None) and (self._end_time is not None):
            return self._end_time - self._start_time

    def start(self):
        self._start_time = time_now()

    def end(self):

        if self._start_time is None:
            raise StoppingTimerBeforeStartingException()

        self._end_time = time_now()

        print(f"Time elapsed for {self._name}: {self.elapsed_time:.3f}s")

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class CumulativeTimer:

    class Entry:

        def __init__(self):
            self.cumulative_time: float = 0
            self.runs: int = 0

        @property
        def time_per_run(self) -> float:
            return self.cumulative_time / self.runs

        def add_run(self, time: float) -> None:
            self.cumulative_time += time
            self.runs += 1

    __timer_vals: Dict[str, Entry] = {}

    def __init__(self, identifier: str):
        self._identifier = identifier
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @property
    def elapsed_time(self) -> Optional[float]:
        if (self._start_time is not None) and (self._end_time is not None):
            return self._end_time - self._start_time

    def start(self):
        self._start_time = time_now()

    def end(self):

        if self._start_time is None:
            raise StoppingTimerBeforeStartingException()

        self._end_time = time_now()

        if self.elapsed_time is not None:
            CumulativeTimer.__add_to_time(self._identifier, self.elapsed_time)
        else:
            raise Exception("Timer elapsed time can't be found")

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    @staticmethod
    def __add_to_time(identifier: str, time: float) -> None:

        if identifier not in CumulativeTimer.__timer_vals:
            CumulativeTimer.__timer_vals[identifier] = CumulativeTimer.Entry()

        CumulativeTimer.__timer_vals[identifier].add_run(time)

    @staticmethod
    def get_time_entry(identifier: str) -> Entry:
        if identifier in CumulativeTimer.__timer_vals:
            return CumulativeTimer.__timer_vals[identifier]
        else:
            return CumulativeTimer.Entry()

    @staticmethod
    def get_all_entries() -> Dict[str, Entry]:
        return CumulativeTimer.__timer_vals.copy()
