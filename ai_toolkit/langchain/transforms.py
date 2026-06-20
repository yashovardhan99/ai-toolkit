"""Package for AI Toolkit LangChain transform integrations."""

from collections.abc import Sequence

from langchain_core.documents import BaseDocumentTransformer
from langchain_core.documents import Document as LangChainDocument

from ai_toolkit import Document
from ai_toolkit.transforms import DocumentTransform

from .types import from_langchain_document, to_langchain_document


class LangChainDocumentTransformer(DocumentTransform):
    """A wrapper for LangChain's BaseDocumentTransformer to be used as a DocumentTransform."""

    def __init__(self, transformer: BaseDocumentTransformer):
        """Initialize the LangChainDocumentTransformer with a LangChain BaseDocumentTransformer.

        Args:
            transformer (BaseDocumentTransformer): The LangChain BaseDocumentTransformer
                to use for transforming.
        """
        self.transformer = transformer

    async def transform(self, documents: Sequence[Document]) -> Sequence[Document]:
        """Transform the content of the given documents using the LangChain BaseDocumentTransformer.

        Args:
            documents (Sequence[Document]): The Document objects to transform.

        Returns:
            Sequence[Document]: A sequence of new Document objects with transformed content.
        """
        langchain_documents = [to_langchain_document(doc) for doc in documents]
        processed_documents = await self.transformer.atransform_documents(
            langchain_documents
        )
        return [from_langchain_document(doc) for doc in processed_documents]


def from_langchain_transformer(
    transformer: BaseDocumentTransformer,
) -> LangChainDocumentTransformer:
    """Create a LangChainDocumentTransformer from a LangChain BaseDocumentTransformer.

    Args:
        transformer (BaseDocumentTransformer): The LangChain BaseDocumentTransformer
            to wrap.

    Returns:
        LangChainDocumentTransformer: An instance of LangChainDocumentTransformer
            that wraps the given LangChain BaseDocumentTransformer.
    """
    return LangChainDocumentTransformer(transformer)


def to_langchain_transformer(transform: DocumentTransform) -> BaseDocumentTransformer:
    """Get the underlying LangChain BaseDocumentTransformer.

    Returns:
        BaseDocumentTransformer: The underlying LangChain BaseDocumentTransformer.
    """
    if isinstance(transform, LangChainDocumentTransformer):
        return transform.transformer

    class _WrapperTransformer(BaseDocumentTransformer):
        """A wrapper for a DocumentTransform to be used as a LangChain BaseDocumentTransformer."""

        async def atransform_documents(
            self,
            documents: Sequence[LangChainDocument],
            **kwargs,
        ) -> Sequence[LangChainDocument]:
            """Transform the content of the given documents using the wrapped DocumentTransform."""
            toolkit_documents = [from_langchain_document(doc) for doc in documents]
            transformed_documents = await transform.transform(toolkit_documents)
            return [to_langchain_document(doc) for doc in transformed_documents]

        def transform_documents(
            self,
            documents: Sequence[LangChainDocument],
            **kwargs,
        ) -> Sequence[LangChainDocument]:
            """Synchronous transformation is not supported.

            This is because the underlying DocumentTransform is asynchronous.
            """
            raise NotImplementedError("Synchronous transformation is not supported.")

    return _WrapperTransformer()
