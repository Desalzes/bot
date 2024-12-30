import time
from collections import deque
from datetime import datetime, timedelta
import logging


class RateLimiter:
    def __init__(self, calls_per_second=1, calls_per_minute=60):
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.second_calls = deque(maxlen=calls_per_second)
        self.minute_calls = deque(maxlen=calls_per_minute)
        self.logger = logging.getLogger(__name__)

    def wait(self):
        """Implements the rate limiting logic"""
        current_time = datetime.now()

        # Clean old entries
        while self.second_calls and current_time - self.second_calls[0] > timedelta(seconds=1):
            self.second_calls.popleft()
        while self.minute_calls and current_time - self.minute_calls[0] > timedelta(minutes=1):
            self.minute_calls.popleft()

        # Check limits
        if len(self.second_calls) >= self.calls_per_second:
            sleep_time = 1 - (current_time - self.second_calls[0]).total_seconds()
            if sleep_time > 0:
                self.logger.debug(f"Rate limit hit (per second). Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

        if len(self.minute_calls) >= self.calls_per_minute:
            sleep_time = 60 - (current_time - self.minute_calls[0]).total_seconds()
            if sleep_time > 0:
                self.logger.debug(f"Rate limit hit (per minute). Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

        # Record the call
        current_time = datetime.now()
        self.second_calls.append(current_time)
        self.minute_calls.append(current_time)
