"""Change finder: self-supervised change detection from multi-year Sentinel-2."""

from .create_windows import create_windows
from .create_windows_urban import create_windows_urban

workflows = {
    "create_windows": create_windows,
    "create_windows_urban": create_windows_urban,
}
