import subprocess
import sys

from app.core.single_instance import InstanceLock, lock_path_for


def test_second_holder_is_refused_until_the_first_releases(tmp_path):
    path = tmp_path / "db.sqlite.lock"
    first, second = InstanceLock(path), InstanceLock(path)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_lock_held_by_another_process_is_refused(tmp_path):
    # The real failure is two PROCESSES; an in-process check alone would not
    # prove the OS lock works across them.
    path = tmp_path / "db.sqlite.lock"
    holder = InstanceLock(path)
    assert holder.acquire()
    code = (
        "import sys; from app.core.single_instance import InstanceLock; "
        f"sys.exit(0 if InstanceLock(r'{path}').acquire() else 3)"
    )
    try:
        assert subprocess.run([sys.executable, "-c", code]).returncode == 3
    finally:
        holder.release()
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_lock_path_sits_beside_the_sqlite_file():
    assert str(lock_path_for("sqlite+aiosqlite:///./cryptoalgo.db")).endswith("cryptoalgo.db.lock")
    assert lock_path_for("sqlite+aiosqlite:////data/live.db").as_posix() == "/data/live.db.lock"
