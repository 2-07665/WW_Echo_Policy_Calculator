"""
OCR helpers for extracting voice-echo upgrade data on Windows builds.

The public API is intentionally small so the Electron bridge can consume it
without worrying about the underlying implementation details.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from .service import EchoUpgradeOCRService, OCRDebugInfo, OCRStatus
    from .workflow import BuffSlotResult, BuffWorkflow, BuffWorkflowResult

    __all__ = [
        "BuffSlotResult",
        "BuffWorkflow",
        "BuffWorkflowResult",
        "EchoUpgradeOCRService",
        "OCRDebugInfo",
        "OCRStatus",
    ]
else:  # pragma: no cover - non-Windows platforms never expose OCR utilities
    __all__: list[str] = []
