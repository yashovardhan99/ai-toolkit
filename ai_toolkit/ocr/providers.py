"""Sample OCR providers."""

import asyncio
from abc import abstractmethod
from collections.abc import Collection, Iterable, Sequence
from typing import Protocol

from ai_toolkit import Document
from ai_toolkit.types import File


class OcrProvider(Protocol):
    """A protocol for OCR providers."""

    @abstractmethod
    async def extract(self, files: Iterable[File]) -> Sequence[Document]:
        """Perform OCR on the given files.

        Args:
            files (Iterable[File]): The files to perform OCR on.

        Returns:
            Sequence[Document]: A sequence of extracted documents.
        """
        raise NotImplementedError("Subclasses must implement the extract method.")

    @abstractmethod
    def get_supported_mime_types(self) -> Collection[str]:
        """Get the supported MIME types for this provider.

        Returns:
            Collection[str]: A collection of supported MIME types.
        """
        raise NotImplementedError(
            "Subclasses must implement the get_supported_mime_types method."
        )


class TextReader(OcrProvider):
    """A provider that reads text from files without performing OCR."""

    async def extract(self, files: Iterable[File]) -> Sequence[Document]:
        """Read text from the given files.

        Args:
            files (Iterable[File]): The files to read text from.

        Returns:
            Sequence[Document]: A sequence of documents with the read text.
        """
        file_contents = await asyncio.gather(
            *(asyncio.to_thread(file.path.read_text) for file in files)
        )
        documents = []
        for file, content in zip(files, file_contents, strict=True):
            documents.append(Document(content=content, source=file.path.as_posix()))
        return documents

    def get_supported_mime_types(self) -> Collection[str]:
        """Get the supported MIME types for this provider."""
        return {"text/plain", "text/markdown"}
