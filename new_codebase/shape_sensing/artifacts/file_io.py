"""Atomic result replacement, including transient Windows sharing violations."""
import os
import time


def atomic_replace(source, target):
    """Retry brief Windows file locks; retain atomic replacement and real errors."""
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))
