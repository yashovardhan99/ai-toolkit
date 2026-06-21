"""OCR package for ai_toolkit."""

from .base import OcrDispatcher, OcrProvider
from .providers import TextReader

__all__ = ["OcrDispatcher", "OcrProvider", "TextReader"]
