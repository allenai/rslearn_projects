"""Shared fixtures for vessel integration tests."""

import functools
from collections.abc import Iterator
from typing import Any

import pytest

import rslp.utils.rslearn as _rslearn_mod


@pytest.fixture(autouse=True)
def _force_serial_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force apply_on_windows to use workers=0 so no multiprocessing pools are spawned."""
    _original = _rslearn_mod.apply_on_windows

    @functools.wraps(_original)
    def _serial(*args: Any, **kwargs: Any) -> Iterator[Any]:
        kwargs["workers"] = 0
        yield from _original(*args, **kwargs)

    monkeypatch.setattr(_rslearn_mod, "apply_on_windows", _serial)
