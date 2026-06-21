"""Llama-Index OCR module."""

import asyncio
from collections.abc import Iterable, Sequence

from liteparse import LiteParse

from ai_toolkit import Document
from ai_toolkit.ocr import OcrProvider
from ai_toolkit.types import File


class LiteParseProvider(OcrProvider):
    """An OCR provider that uses the LiteParse library to extract text from files."""

    def __init__(self, **config):
        """Initialize the LiteParseProvider with an optional configuration."""
        if "output_format" in config:
            if config["output_format"] != "markdown":
                print(
                    "LiteParseProvider only supports 'markdown' output format. "
                    "Overriding to 'markdown'."
                )  # TODO: Add logging instead of print statements
            config.pop("output_format")
        self.parser = LiteParse(output_format="markdown", **config)

    def get_supported_mime_types(self) -> Sequence[str]:
        """Return a sequence of supported MIME types."""
        return ["application/pdf"]

    async def extract(self, files: Iterable[File]) -> Sequence[Document]:
        """Perform OCR on the given files and return a sequence of Documents."""
        tasks = [self._extract_file(file) for file in files]
        return await asyncio.gather(*tasks)

    async def _extract_file(self, file: File) -> Document:
        """Perform OCR on the given file and return a Document."""
        result = await asyncio.to_thread(self.parser.parse, file.path)
        return Document(
            content=result.text,
            metadata={"source": file.path, "_provider": "liteparse"},
        )
