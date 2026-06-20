"""Base classes and utilities for OCR."""

import copy
import itertools
import mimetypes
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Protocol

from ai_toolkit import Document
from ai_toolkit.types import File


class OcrProvider(Protocol):
    """A provider of OCR services."""

    async def extract(self, files: Iterable[File]) -> Sequence[Document]:
        """Perform OCR on the given files.

        Args:
            files (Iterable[File]): The files to perform OCR on.

        Returns:
            Sequence[Document]: A sequence of extracted documents.
        """
        raise NotImplementedError("Subclasses must implement the extract method.")

    def register_with_dispatcher(self, dispatcher: OcrDispatcher) -> None:
        """Register this provider with the given OcrDispatcher."""
        raise NotImplementedError(
            "Subclasses must implement the register_with_dispatcher method."
        )


class OcrDispatcher:
    """A dispatcher for OCR providers based on file MIME types."""

    def __init__(self):
        """Initialize the OcrDispatcher with a sequence of OcrProviders."""
        self.providers: dict[str, list[OcrProvider]] = {}
        self.default_provider: OcrProvider | None = None

    def register_provider(
        self, mime_types: Collection[str], provider: OcrProvider
    ) -> None:
        """Register an OcrProvider for the given MIME types.

        Args:
            mime_types (Collection[str]): The MIME types that the provider supports.
            provider (OcrProvider): The OcrProvider to register.
        """
        for mime_type in mime_types:
            if mime_type not in self.providers:
                self.providers[mime_type] = []
            self.providers[mime_type].append(provider)

    def set_default_provider(self, provider: OcrProvider) -> None:
        """Set the default OcrProvider for all MIME types."""
        self.default_provider = provider

    async def dispatch_folder(
        self, folder_path: str | Path, batch_size: int = 10
    ) -> Sequence[Document]:
        """Perform OCR on all files in the given folder using the appropriate providers.

        Args:
            folder_path (str | Path): The path to the folder containing files to perform OCR on.
            batch_size (int): The number of files to process in each batch.

        Returns:
            Sequence[Document]: A sequence of extracted documents.
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory.")

        files = (
            File(path=file_path)
            for file_path in folder_path.iterdir()
            if file_path.is_file()
        )
        return await self.dispatch(files, batch_size=batch_size)

    async def dispatch(
        self, files: Iterable[File], batch_size: int = 10, maintain_order: bool = True
    ) -> Sequence[Document]:
        """Perform OCR on the given files using the appropriate providers.

        Files with the same mime type are grouped together and processed in batches.
        If `maintain_order` is true, the grouping process maintains the order of the
        files, so the output documents will be in the same order as the input files.
        The grouping only takes place across consecutive files. As such, if the input
        files are not ordered by mime type, the dispatcher will process much smaller
        batches of files, which may result in a less efficient OCR process.

        For optimal performance, it is recommended to either sort the input files by
        mime type or set `maintain_order` to false, which will allow the dispatcher
        to group files by mime type regardless of their order in the input sequence.

        Args:
            files (Iterable[File]): The files to perform OCR on.
            batch_size (int): The number of files to process in each batch.
            maintain_order (bool): Whether to maintain the order of the input files.

        Returns:
            Sequence[Document]: A sequence of extracted documents.
        """
        files_with_mime = (
            copy.replace(file, mime_type=mimetypes.guess_file_type(file.path)[0])
            if file.mime_type is None
            else file
            for file in files
        )
        if not maintain_order:
            files_with_mime = sorted(files_with_mime, key=lambda file: file.mime_type)
        groups = itertools.groupby(files_with_mime, key=lambda file: file.mime_type)
        documents = []
        for mime_type, group in groups:
            batches = itertools.batched(group, batch_size, strict=False)
            for batch in batches:
                batch_docs = await self._dispatch_batch(batch, mime_type)
                documents.extend(batch_docs)
        return documents

    async def _dispatch_batch(self, files: Sequence[File], mime_type: str):
        """Perform OCR on a batch of files with the same MIME type.

        Args:
            files (Sequence[File]): The files to perform OCR on.
            mime_type (str): The MIME type of the files.

        Returns:
            Sequence[Document]: A sequence of extracted documents.
        """
        if mime_type in self.providers:
            return await self.providers[mime_type][0].extract(files)
        elif self.default_provider is not None:
            return await self.default_provider.extract(files)
        else:
            raise ValueError(f"No OCR provider registered for MIME type: {mime_type}")
