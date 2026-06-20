"""Module for defining types used in the AI Toolkit."""

from typing import Any

from attrs import field, frozen


@frozen
class Document:
    """A document."""

    content: str
    metadata: dict[str, Any] = field(factory=dict)
