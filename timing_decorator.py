# timing_decorator.py
"""
Timing utilities for measuring question generation performance.
"""
import time
import logging
from functools import wraps
from typing import Dict, Any, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TimingStats:
    """Store and manage timing statistics."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_time: float = 0
        self.end_time: float = 0
    
    def record(self, stage: str, duration: float):
        """Record timing for a stage."""
        self.timings[stage] = duration
        logger.info(f"⏱️  {stage}: {duration:.2f}s")
    
    def get_total_time(self) -> float:
        """Get total elapsed time."""
        return sum(self.timings.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get formatted timing summary."""
        total = self.get_total_time()
        return {
            "total_time": total,
            "stages": self.timings,
            "total_formatted": format_duration(total),
            "stages_formatted": {
                stage: format_duration(duration)
                for stage, duration in self.timings.items()
            }
        }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


@contextmanager
def time_stage(stats: TimingStats, stage_name: str):
    """Context manager for timing a workflow stage."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        stats.record(stage_name, duration)


def timed_function(stage_name: str):
    """Decorator for timing functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"⏱️  {stage_name}: {format_duration(duration)}")
            return result
        return wrapper
    return decorator