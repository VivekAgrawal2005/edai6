"""
==========================================================
Timer — Processing Time Tracker
==========================================================
Context manager and decorator for measuring execution time.
"""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict, None, None]:
    """
    Context manager that measures elapsed wall-clock time in milliseconds.

    Usage
    -----
    >>> with timer() as t:
    ...     do_work()
    >>> print(t["elapsed_ms"])
    142.35
    """
    result: dict = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result["elapsed_ms"] = round((end - start) * 1000, 2)


def measure_time(func):
    """
    Decorator that adds `processing_time_ms` to the return dict of a function.

    The wrapped function must return a dict. The decorator adds the elapsed
    time under the key ``processing_time_ms``.
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        if isinstance(result, dict):
            result["processing_time_ms"] = elapsed
        return result
    return wrapper
