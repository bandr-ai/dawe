"""DAWE: edit model weights instead of retraining."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("dawe")
except _metadata.PackageNotFoundError:  # running from a bare source tree
    __version__ = "0+unknown"
