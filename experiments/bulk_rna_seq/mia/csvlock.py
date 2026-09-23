"""Atomic, lock-guarded updates to a shared file.

Every results file in this repo is a read-modify-write: load what is there,
merge one row in, write it all back.  That is a race the moment two processes
share it, and it is not a theoretical one -- two experiment processes sharing
`results/index.csv` interleaved and left NUL bytes in it, and every later reader
died with `_csv.Error: line contains NUL`.

The fix is the same everywhere: hold an exclusive lock across the whole
read-modify-write, and install the new version with `os.replace`, which is
atomic on POSIX.  A reader that takes no lock still never sees a torn file --
only the old version or the new one.

    with atomic_update(path) as tmp:
        old = pd.read_csv(path) if path.exists() else empty
        pd.concat([old, new]).to_csv(tmp, index=False)

The lock lives beside the target as `<name>.lock` and is never deleted; an empty
lock file is cheaper than the race that removing it would reintroduce.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_update(path):
    """Lock `path`, yield a temp path to write, then swap it in.

    The body may read `path` freely: the lock is already held, so no other
    holder can be mid-write.  If the body raises, the temp file is discarded and
    `path` is left exactly as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")

    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield tmp_path
            if tmp_path.exists():
                # fsync before the swap, so a crash cannot leave a short file
                # under the target's name.
                with open(tmp_path, "rb+") as f:
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_row(path, row: dict, key: list[str] | None = None) -> None:
    """Append one row to a CSV, de-duplicating on `key`, safely under load.

    Columns are the union across existing and new rows, so a run that records a
    field earlier runs did not will widen the file rather than fail.
    """
    import pandas as pd

    path = Path(path)
    with atomic_update(path) as tmp:
        df = pd.DataFrame([row])
        if path.exists() and path.stat().st_size > 0:
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        if key:
            present = [c for c in key if c in df.columns]
            if present:
                df = df.drop_duplicates(subset=present, keep="last")
        df.to_csv(tmp, index=False)
