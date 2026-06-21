"""Module for defining types used in the AI Toolkit."""

from pathlib import Path
from typing import Any

from attrs import field, frozen


@frozen
class Document:
    """A document."""

    content: str
    metadata: dict[str, Any] = field(factory=dict)

    @property
    def source(self) -> str | None:
        """Get the source of the document from its metadata."""
        return self.metadata.get("source")


def _ensure_normalized_path(path: str | Path) -> Path:
    """Ensure that the path is a normalized Path object."""
    if isinstance(path, str):
        return Path(path).resolve()
    else:
        return path.resolve()


@frozen
class File:
    """A file."""

    path: Path = field(converter=_ensure_normalized_path)
    mime_type: str | None = None
