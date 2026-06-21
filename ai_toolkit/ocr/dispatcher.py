"""Base classes and utilities for OCR."""

import itertools
import mimetypes
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path

import attrs

from ai_toolkit import Document
from ai_toolkit.types import File

from .providers import OcrProvider


class OcrDispatcher:
    """A dispatcher for OCR providers based on file MIME types."""

    def __init__(self, providers: Sequence[OcrProvider] | None = None):
        """Initialize the OcrDispatcher with a sequence of OcrProviders."""
        self.providers: dict[str, list[OcrProvider]] = {}
        self.default_provider: OcrProvider | None = None
        if providers:
            for provider in providers:
                self.register_provider(provider)

    def register_provider(self, provider: OcrProvider) -> None:
        """Register an OcrProvider for the given MIME types.

        Args:
            provider (OcrProvider): The OcrProvider to register.
        """
        for mime_type in provider.get_supported_mime_types():
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
        documents = []
        async for batch in self.stream_dispatch(
            files, batch_size=batch_size, maintain_order=maintain_order
        ):
            documents.extend(batch)
        return documents

    async def stream_dispatch(
        self, files: Iterable[File], batch_size: int = 10, maintain_order: bool = True
    ) -> AsyncIterator[Sequence[Document]]:
        """Perform OCR on the given files using the appropriate providers, streaming results back.

        This method is similar to `dispatch`, but it yields documents as they are processed instead
        of returning a complete sequence at the end. This allows for more efficient memory usage and
        faster processing of large batches of files.

        Args:
            files (Iterable[File]): The files to perform OCR on.
            batch_size (int): The number of files to process in each batch.
            maintain_order (bool): Whether to maintain the order of the input files.

        Yields:
            Sequence[Document]: A sequence of batches as they are processed.
        """
        files_with_mime = (
            attrs.evolve(file, mime_type=mimetypes.guess_type(file.path)[0])
            if file.mime_type is None
            else file
            for file in files
        )
        files_with_providers = (
            (file, self._get_provider(file.mime_type)) for file in files_with_mime
        )
        if not maintain_order:
            files_with_providers = sorted(
                files_with_providers, key=lambda pair: pair[0].mime_type
            )
        groups: itertools.groupby[OcrProvider, tuple[File, OcrProvider]] = (
            itertools.groupby(files_with_providers, key=lambda pair: pair[1])
        )
        for provider, group in groups:
            batches = itertools.batched(group, batch_size)
            for batch in batches:
                yield await provider.extract([file for file, _ in batch])

    def _get_provider(self, mime_type: str | None) -> OcrProvider:
        """Get the OcrProvider for the given MIME type.

        Args:
            mime_type (str): The MIME type to get the provider for.

        Returns:
            OcrProvider: The OcrProvider for the given MIME type, or the default provider
                if no specific provider is registered.

        Raises:
            ValueError: If no provider is registered for the given MIME type and no default
                provider is set.
        """
        if mime_type in self.providers:
            return self.providers[mime_type][0]
        if self.default_provider is not None:
            return self.default_provider
        raise ValueError(f"No OCR provider registered for MIME type: {mime_type}")

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
