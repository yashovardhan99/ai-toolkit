"""Type conversions between AI Toolkit and LangChain."""

from langchain_core.documents import Document as LangChainDocument

from ai_toolkit import Document


def from_langchain_document(document: LangChainDocument) -> Document:
    """Convert a LangChain Document to an AI Toolkit Document."""
    return Document(content=document.page_content, metadata=document.metadata)


def to_langchain_document(document: Document) -> LangChainDocument:
    """Convert an AI Toolkit Document to a LangChain Document."""
    return LangChainDocument(page_content=document.content, metadata=document.metadata)
