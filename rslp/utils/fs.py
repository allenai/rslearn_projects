"""fsspec-related utilities."""

import os
import shutil
from typing import Any

import fsspec
import upath.registry
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem
from upath import UPath
from upath._flavour import WrappedFileSystemFlavour
from upath.implementations.cloud import CloudPath


def copy_file(src_fname: UPath, dst_fname: UPath) -> None:
    """Copy a file from src to dst.

    This is mainly for use with multiprocessing.
    """
    if isinstance(dst_fname.fs, LocalFileSystem):
        # When copying to local filesystem, we should ensure parent directory is
        # created.
        dst_fname.parent.mkdir(parents=True, exist_ok=True)
    with src_fname.open("rb") as src:
        with dst_fname.open("wb") as dst:
            shutil.copyfileobj(src, dst)


class WekaFileSystem(S3FileSystem):
    """fsspec FileSystem implementation for Weka.

    This way we can still provide keys through environment variables, but use different
    environment variables for Weka.
    """

    protocol = ("weka",)

    def __init__(self, **kwargs: dict[str, Any]):
        """Create a new WekaFileSystem.

        Args:
            kwargs: see S3FileSystem.
        """
        super().__init__(
            key=os.environ["WEKA_ACCESS_KEY_ID"],
            secret=os.environ["WEKA_SECRET_ACCESS_KEY"],
            endpoint_url=os.environ["WEKA_ENDPOINT_URL"],
            **kwargs,
        )


class WekaPath(CloudPath):
    """UPath implementation for Weka."""

    __slots__ = ()

    def __init__(
        self, *args: list[Any], protocol: str | None = None, **storage_options: Any
    ) -> None:
        """Create a new WekaPath.

        Args:
            args: see CloudPath.
            protocol: the protocol name, should be "weka".
            storage_options: filesystem options.
        """
        super().__init__(*args, protocol=protocol, **storage_options)
        if not self.drive and len(self.parts) > 1:
            raise ValueError("non key-like path provided (bucket/container missing)")


fsspec.register_implementation("weka", WekaFileSystem)
upath.registry.register_implementation("weka", WekaPath)
WrappedFileSystemFlavour.protocol_config["netloc_is_anchor"].add("weka")
WrappedFileSystemFlavour.protocol_config["supports_empty_parts"].add("weka")
