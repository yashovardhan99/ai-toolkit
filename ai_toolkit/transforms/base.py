"""Base classes and utilities for document transforms."""

import asyncio
import copy
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterable, Sequence
from itertools import batched
from typing import Protocol

from ai_toolkit import Document


class DocumentTransform(Protocol):
    """A transformation that can be applied to Document objects."""

    @abstractmethod
    async def transform(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Transform the given Document objects.

        Args:
            documents (Sequence[Document]): The Document objects to transform.

        Returns:
            Sequence[Document]: The transformed Document objects.
        """
        raise NotImplementedError("Subclasses must implement the transform method.")


class MetadataExtractor(DocumentTransform, Protocol):
    """A transformation that extracts metadata from Document objects."""

    async def transform(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Extract metadata from the given Document objects.

        Args:
            documents (Sequence[Document]): The Document objects to extract metadata from.

        Returns:
            Sequence[Document]: The Document objects with extracted metadata applied.
        """
        metadata_list = await self.extract(documents)
        return [
            copy.replace(document, metadata=document.metadata | metadata)
            for document, metadata in zip(documents, metadata_list, strict=True)
        ]

    @abstractmethod
    async def extract(self, documents: Sequence[Document]) -> Sequence[dict]:
        """Extract metadata from the given Document objects.

        Args:
            documents (Sequence[Document]): The Document objects to extract metadata from.

        Returns:
            Sequence[dict]: A sequence of metadata dictionaries extracted from the documents.
        """
        raise NotImplementedError("Subclasses must implement the extract method.")


class SyncDocumentTransform(DocumentTransform, Protocol):
    """A synchronous transformation that can be applied to Document objects."""

    @abstractmethod
    def transform_sync(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Transform the given Document objects.

        Args:
            documents (Sequence[Document]): The Document objects to transform.

        Returns:
            Sequence[Document]: The transformed Document objects.
        """
        raise NotImplementedError(
            "Subclasses must implement the transform_sync method."
        )

    async def transform(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Transform the given Document objects asynchronously.

        Args:
            documents (Sequence[Document]): The Document objects to transform.

        Returns:
            Sequence[Document]: The transformed Document objects.
        """
        return await asyncio.to_thread(self.transform_sync, documents)


class TransformPipeline:
    """Pipeline for applying document transforms."""

    def __init__(self, transforms: Iterable[DocumentTransform] | None = None):
        """Initialize the TransformPipeline."""
        self.transforms: list[DocumentTransform] = []
        if transforms is not None:
            self.transforms.extend(transforms)

    def add_transform(self, transform: DocumentTransform) -> None:
        """Add a transform to the pipeline.

        Args:
            transform (DocumentTransform): The transform to add.
        """
        self.transforms.append(transform)

    async def run(
        self,
        documents: Iterable[Document],
        batch_size: int = 10,
    ) -> AsyncIterator[Document]:
        """Apply registered transforms to the given documents.

        Args:
            documents (Iterable[Document]): An iterable of Document objects to transform.
            batch_size (int): The number of documents to process in each batch
                when applying document transforms. If a transform produces a different
                number of documents than it consumes, the next transform will receive the
                documents produced by the previous transform, regardless of the batch size.

        Yields:
            Document: Transformed Document objects.
        """
        for batch in batched(documents, batch_size, strict=False):
            for transform in self.transforms:
                batch = await transform.transform(batch)
            for doc in batch:
                yield doc

    async def run_and_collect(
        self,
        documents: Iterable[Document],
        batch_size: int = 10,
    ) -> Sequence[Document]:
        """Apply registered transforms to the given documents and collect results.

        Args:
            documents (Iterable[Document]): An iterable of Document objects to transform.
            batch_size (int): The number of documents to process in each batch
                when applying document transforms. If a transform produces a different
                number of documents than it consumes, the next transform will receive the
                documents produced by the previous transform, regardless of the batch size.

        Returns:
            Sequence[Document]: A list of transformed Document objects.
        """
        return [doc async for doc in self.run(documents, batch_size)]


class TransformRegistry:
    """Registry for managing document transforms."""

    def __init__(self):
        """Initialize the TransformRegistry."""
        self._registry: dict[str, DocumentTransform] = {}

    def register(self, name: str, transform: DocumentTransform) -> None:
        """Register a transform with the given name.

        Args:
            name (str): The name of the transform.
            transform (DocumentTransform): The transform to register.
        """
        self._registry[name] = transform

    def get(self, name: str) -> DocumentTransform:
        """Get a registered transform by name.

        Args:
            name (str): The name of the transform.

        Returns:
            DocumentTransform: The registered transform.

        Raises:
            KeyError: If the transform is not found in the registry.
        """
        return self._registry[name]

    def build_pipeline(self, names: Iterable[str]) -> TransformPipeline:
        """Build a TransformPipeline from the registered transforms.

        Args:
            names (Iterable[str]): An iterable of transform names to include in the pipeline.

        Returns:
                TransformPipeline: A pipeline containing the specified transforms.
        """
        transforms = (self.get(name) for name in names)
        return TransformPipeline(transforms)
