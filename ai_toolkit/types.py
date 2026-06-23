"""Module for defining types used in the AI Toolkit."""

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attrs import Converter, field, frozen


def _set_default_metadata(metadata: Mapping[str, Any], self_) -> Mapping[str, Any]:
    """Set default metadata values for a Document object."""
    metadata_dict = dict(metadata)

    metadata_dict["id"] = self_.id_
    if self_.source is not None:
        metadata_dict["source"] = self_.source
    return metadata_dict


@frozen
class Document:
    """A document.

    If an `id_` is not set, a random UUID will be generated instead.
    Both `id_` and `source` will be added to the metadata dictionary.

    Attributes:
        content (str): The content of the document.
        id_ (str): The ID of the document.
        source (str | None): The source of the document, if available.
        metadata (Mapping[str, Any]): The metadata of the document.
    """

    content: str
    id_: str = field(
        factory=lambda: str(uuid.uuid4()),
    )
    source: str | None = field(default=None)
    metadata: Mapping[str, Any] = field(
        factory=dict, converter=Converter(_set_default_metadata, takes_self=True)
    )


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
