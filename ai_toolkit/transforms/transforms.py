"""Ready-to-use transforms for AI Toolkit."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from ai_toolkit import Document

from .base import DocumentTransform


class SimpleDirectoryWriter(DocumentTransform):
    """A transform that saves Document objects to a specified directory.

    This transform saves Document objects to a specified directory. Optionally,
    it can also persist metadata if the `persist_metadata` flag is set to True.

    Documents are saved as .txt files with their IDs as filenames. If `persist_metadata`
    is True, metadata is saved in accompanying JSON files with the same name as
    the document file, but with a `_metadata.json` suffix.
    """

    def __init__(
        self,
        output_dir: str | Path,
        persist_metadata: bool = False,
    ) -> None:
        """Initialize the SimpleDirectoryWriter.

        Args:
            output_dir (str | Path): The directory where the documents will be saved.
            persist_metadata (bool): Whether to persist document metadata.
        """
        self.output_dir: Path = Path(output_dir).resolve()
        self.persist_metadata = persist_metadata
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def transform(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Save the given Document objects to the specified directory.

        Args:
            documents (Sequence[Document]): The Document objects to save.

        Returns:
            Sequence[Document]: The saved Document objects.
        """
        await asyncio.gather(*(self._save_document(document) for document in documents))
        return documents

    async def save(self, documents: Sequence[Document]) -> None:
        """Save the given Document objects to the specified directory."""
        await self.transform(documents)

    async def _save_document(self, document: Document) -> None:
        """Save a single Document object to the specified directory.

        Args:
            document (Document): The Document object to save.
        """
        file_path = self.output_dir / f"{document.id_}.txt"
        await asyncio.to_thread(
            file_path.write_text, document.content, encoding="utf-8"
        )
        if self.persist_metadata:
            metadata_path = self.output_dir / f"{document.id_}_metadata.json"
            await asyncio.to_thread(
                json.dump,
                document.metadata,
                metadata_path.open("w", encoding="utf-8"),
            )


class SimpleDirectoryReader(DocumentTransform):
    """A transform that reads Document objects from a specified directory.

    This transform reads .txt and .md files from the specified directory and creates
    Document objects from them. Optionally, it can also read metadata from accompanying
    JSON files if the `load_metadata` flag is set to True. This transform can be used in
    conjunction with the SimpleDirectoryWriter to read documents that were previously saved.
    """

    def __init__(
        self,
        input_dir: str | Path,
        file_name_as_id: bool = True,
        load_metadata: bool = False,
    ) -> None:
        """Initialize the SimpleDirectoryReader.

        Args:
            input_dir (str | Path): The directory from which the documents will be read.
            file_name_as_id (bool): Whether to use the file name as the document ID.
            load_metadata (bool): Whether to load metadata from accompanying JSON files.
        """
        self.input_dir: Path = Path(input_dir).resolve()
        self.file_name_as_id = file_name_as_id
        self.load_metadata = load_metadata

    async def transform(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Read Document objects from the specified directory.

        Args:
            documents (Sequence[Document]): Initially empty sequence of Document objects.
                This parameter is not used in this transform, but is included to conform
                to the DocumentTransform interface.

        Returns:
            Sequence[Document]: The read Document objects.
        """
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise ValueError(
                f"Input directory {self.input_dir} does not exist or is not a directory."
            )
        document_files = await asyncio.to_thread(
            lambda: (
                list(self.input_dir.glob("*.txt")) + list(self.input_dir.glob("*.md"))
            )
        )
        read_documents = await asyncio.gather(
            *(self._read_document(file_path) for file_path in document_files)
        )
        return read_documents

    async def load(self) -> Sequence[Document]:
        """Load Document objects from the specified directory."""
        return await self.transform([])

    def _read_metadata(self, file_path: Path) -> dict:
        """Read metadata from a JSON file corresponding to the given document file.

        Args:
            file_path (Path): The path to the document file.

        Returns:
            dict: The metadata read from the JSON file, or an empty dictionary
                if the file does not exist.
        """
        metadata_path = file_path.with_name(f"{file_path.stem}_metadata.json")
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as f:
                return json.load(f)
        return {}

    async def _read_document(self, file_path: Path) -> Document:
        """Read a single Document object from the specified file path.

        Args:
            file_path (Path): The path to the file to read.

        Returns:
            Document: The read Document object.
        """
        content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        if self.load_metadata:
            metadata = await asyncio.to_thread(self._read_metadata, file_path)
        else:
            metadata = {}
        return Document(
            content=content,
            id_=file_path.stem,
            source=file_path.as_posix(),
            metadata=metadata,
        )
