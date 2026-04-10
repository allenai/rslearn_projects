"""Vessel attribute prediction model."""

from .create_windows import create_windows

workflows = {
    "create_windows": create_windows,
}
