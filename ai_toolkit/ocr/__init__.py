"""OCR package for ai_toolkit."""

from .dispatcher import OcrDispatcher
from .providers import OcrProvider, TextReader

__all__ = ["OcrDispatcher", "OcrProvider", "TextReader"]
