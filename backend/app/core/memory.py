from __future__ import annotations

import ctypes
import sys

# Gives freed heap memory back to the operating system.
#
# CPython + pandas free memory to glibc's allocator, which keeps it for reuse
# instead of returning it, so a long-lived process's resident size only ever
# ratchets up after each allocation-heavy job (the gold research tick rebuilds
# a feature frame every 3 minutes). On a 2 GB server that headroom matters;
# malloc_trim(0) hands the unused part back. A no-op off glibc Linux.

_trim = None
if sys.platform.startswith("linux"):
    try:
        _trim = ctypes.CDLL("libc.so.6").malloc_trim
        _trim.argtypes = [ctypes.c_size_t]
        _trim.restype = ctypes.c_int
    except OSError:
        _trim = None


def release_free_memory() -> bool:
    """Returns True if memory was handed back to the OS."""
    if _trim is None:
        return False
    return bool(_trim(0))
