"""
Phase 5 — AI Detection Tests (Cheel Nazar)
==========================================

Tests for POST /api/v1/ai/detect using FastAPI's TestClient (no server needed).

Run:
    cd backend
    python -m pytest tests/ -v

⚠ These tests use the MockDetector — no YOLO weights are required.
"""

from __future__ import annotations

import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

DETECT_URL = "/api/v1/ai/detect"


# ---------------------------------------------------------------------------
# Helpers — create minimal valid/invalid image bytes in-memory
# ---------------------------------------------------------------------------

def _make_minimal_png(width: int = 4, height: int = 4) -> bytes:
    """
    Build a valid minimal PNG from scratch (no PIL required).
    Uses a 1×1 greyscale PNG encoded by hand so the test has zero heavy deps.
    """
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk: width=1, height=1, 8-bit greyscale, no interlace
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)

    # IDAT chunk: a single black pixel (filter byte 0x00 + one 0x00 pixel byte)
    raw_row = b"\x00\x00"  # filter=None + pixel=0
    compressed = zlib.compress(raw_row)
    idat = png_chunk(b"IDAT", compressed)

    # IEND chunk
    iend = png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Test 1: Valid PNG image — expect HTTP 200 with proper structure
# ---------------------------------------------------------------------------

def test_valid_png_returns_success():
    png_bytes = _make_minimal_png()
    response = client.post(
        DETECT_URL,
        files={"file": ("test_frame.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # Top-level structure
    assert body["status"] == "success"
    assert "model" in body
    assert "image_filename" in body
    assert body["image_filename"] == "test_frame.png"
    assert isinstance(body["detections"], list)


# ---------------------------------------------------------------------------
# Test 2: MockDetector returns expected dummy detection
# ---------------------------------------------------------------------------

def test_mock_detector_returns_dummy_detection():
    png_bytes = _make_minimal_png()
    response = client.post(
        DETECT_URL,
        files={"file": ("drone_frame.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    detections = response.json()["detections"]

    # MockDetector always returns exactly one dummy detection
    assert len(detections) == 1
    det = detections[0]

    assert det["class_name"] == "mock_vehicle"
    assert 0.0 <= det["confidence"] <= 1.0
    assert det["is_mock"] is True

    bbox = det["bbox"]
    for key in ("x1", "y1", "x2", "y2"):
        assert key in bbox


# ---------------------------------------------------------------------------
# Test 3: JPEG extension accepted
# ---------------------------------------------------------------------------

def test_jpeg_extension_accepted():
    # Send PNG bytes but name it .jpg (extension check only — MockDetector doesn't
    # truly decode image format).
    png_bytes = _make_minimal_png()
    response = client.post(
        DETECT_URL,
        files={"file": ("snapshot.jpg", io.BytesIO(png_bytes), "image/jpeg")},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Test 4: Unsupported extension rejected with HTTP 400
# ---------------------------------------------------------------------------

def test_invalid_extension_rejected():
    response = client.post(
        DETECT_URL,
        files={"file": ("document.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 5: MP4 video extension rejected (not an image)
# ---------------------------------------------------------------------------

def test_video_extension_rejected():
    response = client.post(
        DETECT_URL,
        files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test 6: Empty file rejected with HTTP 400
# ---------------------------------------------------------------------------

def test_empty_file_rejected():
    response = client.post(
        DETECT_URL,
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 7: Missing file field returns HTTP 422 (FastAPI validation)
# ---------------------------------------------------------------------------

def test_missing_file_returns_422():
    response = client.post(DETECT_URL)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 8: Response always has the required top-level keys
# ---------------------------------------------------------------------------

def test_response_structure():
    png_bytes = _make_minimal_png()
    response = client.post(
        DETECT_URL,
        files={"file": ("frame.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    for key in ("status", "model", "image_filename", "detections"):
        assert key in body, f"Missing key: {key}"
