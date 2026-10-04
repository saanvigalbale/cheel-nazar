"""
AI Detection Service — Phase 5 (Cheel Nazar)
=============================================

Architecture
------------
This module defines:

  BaseDetector   — abstract interface every detector implementation must satisfy.
  BoundingBox    — typed data structure for bounding box coordinates.
  Detection      — typed data structure for a single detection result.
  MockDetector   — placeholder implementation (returns deterministic dummy data)
                   used to verify the API plumbing WITHOUT downloading any model
                   weights or requiring Ultralytics / PyTorch.

Integration path
----------------
Phase 6 will replace `MockDetector` with `YOLODetector(BaseDetector)` that wraps
Ultralytics YOLO.  The FastAPI endpoint (`/api/v1/ai/detect`) and all callers will
remain unchanged because they depend only on `BaseDetector`.

⚠ NO model weights are downloaded or loaded in this file.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data models (plain dataclasses — zero external dependencies)
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Detection:
    """Single object detection result."""
    class_name: str
    confidence: float          # 0.0 – 1.0
    bbox: BoundingBox
    # Optional metadata for downstream consumers
    is_mock: bool = False      # True when produced by MockDetector


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """
    Contract that every AI detector must fulfil.

    A detector is intentionally kept independent from FastAPI so it can be
    unit-tested, benchmarked, or swapped without touching the HTTP layer.
    """

    @abstractmethod
    def detect(self, image_path: str) -> List[Detection]:
        """
        Run inference on an image file.

        Parameters
        ----------
        image_path : str
            Absolute path to a readable image file (.jpg, .jpeg, .png, .bmp, .tiff).

        Returns
        -------
        List[Detection]
            Possibly-empty list of detection results.

        Raises
        ------
        FileNotFoundError
            If ``image_path`` does not exist.
        ValueError
            If the file cannot be read as an image.
        RuntimeError
            If the underlying model fails unexpectedly.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the underlying model (used in API responses)."""
        ...


# ---------------------------------------------------------------------------
# Mock implementation  ⚠ PLACEHOLDER — no real inference is performed ⚠
# ---------------------------------------------------------------------------

class MockDetector(BaseDetector):
    """
    ⚠ MOCK DETECTOR — FOR ARCHITECTURE VALIDATION ONLY ⚠

    Returns a single deterministic dummy detection so the full API → service
    round-trip can be tested without any ML libraries.

    Replace this class with `YOLODetector` in Phase 6 when Ultralytics is
    available in the environment.

    Dummy detection payload
    -----------------------
    class_name : "mock_vehicle"
    confidence : 0.87
    bbox       : x1=10, y1=20, x2=80, y2=60
    is_mock    : True   ← always set so clients know this is fake
    """

    @property
    def model_name(self) -> str:
        return "MockDetector-v0 (placeholder)"

    def detect(self, image_path: str) -> List[Detection]:
        """
        Validate that the image file exists and is non-empty, then return
        a fixed dummy detection list.  No actual model inference occurs.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if os.path.getsize(image_path) == 0:
            raise ValueError(f"Image file is empty (0 bytes): {image_path}")

        # ── MOCK RESULT ────────────────────────────────────────────────────
        # ⚠ The detection below is fabricated test data.
        # Phase 6 will replace this with real YOLO inference output.
        # ──────────────────────────────────────────────────────────────────
        return [
            Detection(
                class_name="mock_vehicle",
                confidence=0.87,
                bbox=BoundingBox(x1=10.0, y1=20.0, x2=80.0, y2=60.0),
                is_mock=True,
            )
        ]


# ---------------------------------------------------------------------------
# Singleton — injected into FastAPI endpoint via module-level instance
# ---------------------------------------------------------------------------

#: Active detector instance used by the API layer.
#: Swap this to `YOLODetector(...)` in Phase 6 without touching the endpoint.
detector: BaseDetector = MockDetector()
