"""Example usage of the TransformRegistry."""

import asyncio
from collections.abc import Sequence

from ai_toolkit import Document
from ai_toolkit.transforms import (
    DocumentTransform,
    MetadataExtractor,
    TransformPipeline,
    TransformRegistry,
)


class ExampleMetadataExtractor(MetadataExtractor):
    """Example implementation of a MetadataExtractor."""

    async def extract(self, documents: Sequence[Document]) -> Sequence[dict]:
        """Example transformation that adds a new key to the metadata."""
        return [{"example_key": "example_value"} for _ in documents]


class ExampleDocumentTransform(DocumentTransform):
    """Example implementation of a DocumentTransform."""

    def __init__(self, suffix: str = " - transformed"):
        """Initialize the ExampleDocumentTransform with a suffix to append to document content."""
        self.suffix = suffix

    async def transform(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Example transformation that appends text to the document content."""
        new_documents = []
        await asyncio.sleep(0.1)  # Simulate async work
        for document in documents:
            new_content = document.content + self.suffix
            new_documents.append(
                Document(content=new_content, metadata=document.metadata)
            )
        return new_documents


async def main():
    """Example usage of the TransformRegistry and TransformPipeline."""
    docs = [Document(str(i)) for i in range(100)]
    registry = TransformRegistry()
    registry.register("example_metadata_extractor", ExampleMetadataExtractor())
    registry.register(
        "example_document_transform", ExampleDocumentTransform(suffix=" - transformed")
    )
    pipeline = TransformPipeline(
        transforms=[
            ExampleMetadataExtractor(),
            ExampleDocumentTransform(suffix=" - transformed"),
        ]
    )
    pipeline2 = registry.build_pipeline(
        ["example_metadata_extractor", "example_document_transform"]
    )
    docs1 = []
    task2 = pipeline2.run_and_collect(docs)
    async for transformed_doc in pipeline.run(docs):
        docs1.append(transformed_doc)
        print(transformed_doc)
    print("Pipeline 1 complete.")
    docs2 = await task2
    print("Pipeline 2 complete.")
    print(docs1 == docs2)


if __name__ == "__main__":
    asyncio.run(main())
