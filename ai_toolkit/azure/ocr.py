"""Azure OCR module for AI Toolkit.

Provides functions to interact with Azure Document Intelligence
"""

import asyncio
from collections.abc import Collection, Iterable, Sequence

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
)

from ai_toolkit import Document
from ai_toolkit.ocr import OcrProvider
from ai_toolkit.types import File


class AzureDocumentIntelligenceProvider(OcrProvider):
    """Azure Document Intelligence provider for OCR."""

    def __init__(
        self, client: DocumentIntelligenceClient, model_id: str = "prebuilt-layout"
    ):
        """Initialize the Azure Document Intelligence provider.

        Args:
            client (DocumentIntelligenceClient): The Azure Document Intelligence client.
            model_id (str): The model ID to use for document analysis.

        """
        self.client = client
        self.model_id = model_id
        self._supported_mime_types = {
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/bmp",
            "image/tiff",
            "image/heif",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/html",
        }

    async def extract(self, files: Iterable[File]) -> Sequence[Document]:
        """Perform OCR on the given files using Azure Document Intelligence.

        Args:
            files (Iterable[File]): The files to perform OCR on.

        Returns:
            Sequence[Document]: A sequence of extracted documents.
        """
        return await asyncio.gather(*(self._extract_file(file) for file in files))

    async def _extract_file(self, file: File) -> Document:
        """Perform OCR on a single file using Azure Document Intelligence.

        Args:
            file (File): The file to perform OCR on.

        Returns:
            Document: The extracted document.
        """
        request = AnalyzeDocumentRequest(
            bytes_source=await asyncio.to_thread(file.path.read_bytes)
        )
        poller = await asyncio.to_thread(
            self.client.begin_analyze_document,
            self.model_id,
            request,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = await asyncio.to_thread(poller.result)
        return Document(
            content=result.content,
            source=file.path.as_posix(),
            metadata={"_provider": "azure_document_intelligence"},
        )

    def override_supported_mime_types(self, mime_types: set[str]) -> None:
        """Override the supported MIME types for this provider.

        Args:
            mime_types (set[str]): The new set of supported MIME types.
        """
        self._supported_mime_types = mime_types

    def get_supported_mime_types(self) -> Collection[str]:
        """Get the supported MIME types for this provider.

        Returns:
            Collection[str]: The collection of supported MIME types.
        """
        return self._supported_mime_types
