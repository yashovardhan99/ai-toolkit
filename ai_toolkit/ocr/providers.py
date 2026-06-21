"""Sample OCR providers."""

import asyncio
from collections.abc import Iterable, Sequence

from ai_toolkit import Document
from ai_toolkit.ocr.base import OcrDispatcher, OcrProvider
from ai_toolkit.types import File


class TextReader(OcrProvider):
    """A provider that reads text from files without performing OCR."""

    SUPPORTED_MIME_TYPES = {"text/plain", "text/markdown"}

    async def extract(self, files: Iterable[File]) -> Sequence[Document]:
        """Read text from the given files.

        Args:
            files (Iterable[File]): The files to read text from.

        Returns:
            Sequence[Document]: A sequence of documents with the read text.
        """
        documents = []
        for file in files:
            content = await asyncio.to_thread(self._read_file, file)
            documents.append(
                Document(content=content, metadata={"source": file.path.as_posix()})
            )
        return documents

    def _read_file(self, file: File) -> str:
        return file.path.read_text()

    def register_with_dispatcher(
        self, dispatcher: OcrDispatcher, set_default: bool = False
    ) -> None:
        """Register this provider with the given OcrDispatcher."""
        dispatcher.register_provider(self.SUPPORTED_MIME_TYPES, self)
        if set_default:
            dispatcher.set_default_provider(self)
