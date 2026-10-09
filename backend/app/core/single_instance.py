from __future__ import annotations

import os
from pathlib import Path

# One trading process per database, enforced by the OS.
#
# Two backends on the same database each run the scheduler, so each places its
# own orders and records its own equity snapshots against one shared account.
# That is not hypothetical here: the paper equity on 2026-08-19..30 alternates
# by $300-800 between readings a minute apart -- two processes marking the same
# book differently. Running 24/7 from a startup script makes it MORE likely
# (the script's copy plus one started by hand from a terminal), so the guard
# has to live in the app, not in the script.
#
# An OS file lock rather than a PID file: the OS drops it the moment the
# process dies, however it dies, so a crash can never leave a stale lock that
# blocks the next start.


class InstanceLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> bool:
        """True if this process now holds the lock; False if another does."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def lock_path_for(database_url: str) -> Path:
    """`<database file>.lock` beside a SQLite database, so the test database and
    the live one never contend; a non-file database locks in the working dir."""
    if database_url.startswith("sqlite"):
        return Path(database_url.split("///", 1)[-1] + ".lock")
    return Path("cryptoalgo.instance.lock")
