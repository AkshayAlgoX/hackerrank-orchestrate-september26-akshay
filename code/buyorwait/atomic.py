"""Crash-safe file replacement: write a temp file next to the target, then os.replace().

A reader never observes a half-written artefact: until the final rename the destination is
the previous complete file (or absent), and the temp file - a random ``.<name>.<rand>.tmp``
sibling that no consumer looks for - is removed again on any exception.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, IO, Optional


def atomic_write(path: str, write: Callable[[IO], None], mode: str = "w", encoding: Optional[str] = "utf-8",
                 newline: Optional[str] = None, replace: Callable[[str, str], None] = os.replace) -> None:
    """Call ``write(fh)`` on a temp file in ``path``'s directory, then atomically move it to ``path``.

    * The temp file is created by ``tempfile`` in the same directory (same filesystem, so the
      rename is atomic) with a random suffix, so concurrent runs cannot collide.
    * The handle is flushed and fsync'd before it is closed; only then is it renamed.
    * If ``write`` or the rename raises, the temp file is unlinked and the previous ``path`` is
      left untouched; the exception propagates.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    base = os.path.basename(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=directory)
    try:
        kwargs = {} if "b" in mode else {"encoding": encoding, "newline": newline}
        with os.fdopen(fd, mode, **kwargs) as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def is_temp_artifact(name: str) -> bool:
    """True for the sibling temp names this module creates (never a valid artefact)."""
    return name.startswith(".") and name.endswith(".tmp")
