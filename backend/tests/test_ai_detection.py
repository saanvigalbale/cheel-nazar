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
# Helpers — create minimal valid PNG bytes in-memory (no PIL required)
# ---------------------------------------------------------------------------

def _make_minimal_png() -> bytes:
    """
    Hand-craft a 1×1 greyscale PNG from raw bytes.
    Zero external dependencies — works with the stdlib alone.
    """
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=1, height=1, bit_depth=8, colour_type=0 (greyscale)
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)
    # IDAT: filter byte 0x00 + single pixel byte 0x00 (black)
    idat = png_chunk(b"IDAT", zlib.compress(b"\x00\x00"))
    iend = png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Test 1: Valid PNG → HTTP 200 with correct structure
# ---------------------------------------------------------------------------

def test_valid_png_returns_success():
    response = client.post(
        DETECT_URL,
        files={"file": ("test_frame.png", io.BytesIO(_make_minimal_png()), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert "model" in body
    assert body["image_filename"] == "test_frame.png"
    assert isinstance(body["detections"], list)


# ---------------------------------------------------------------------------
# Test 2: MockDetector returns expected dummy detection payload
# ---------------------------------------------------------------------------

def test_mock_detector_returns_dummy_detection():
    response = client.post(
        DETECT_URL,
        files={"file": ("drone_frame.png", io.BytesIO(_make_minimal_png()), "image/png")},
    )
    assert response.status_code == 200
    detections = response.json()["detections"]
    assert len(detections) == 1

    det = detections[0]
    assert det["class_name"] == "mock_vehicle"
    assert 0.0 <= det["confidence"] <= 1.0
    assert det["is_mock"] is True

    for key in ("x1", "y1", "x2", "y2"):
        assert key in det["bbox"]


# ---------------------------------------------------------------------------
# Test 3: JPEG extension is accepted
# ---------------------------------------------------------------------------

def test_jpeg_extension_accepted():
    response = client.post(
        DETECT_URL,
        files={"file": ("snapshot.jpg", io.BytesIO(_make_minimal_png()), "image/jpeg")},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Test 4: .txt extension rejected with HTTP 400
# ---------------------------------------------------------------------------

def test_invalid_extension_rejected():
    response = client.post(
        DETECT_URL,
        files={"file": ("document.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 5: .mp4 extension (video) rejected — not an allowed image format
# ---------------------------------------------------------------------------

def test_video_extension_rejected():
    response = client.post(
        DETECT_URL,
        files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 64), "video/mp4")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test 6: Empty file body rejected with HTTP 400
# ---------------------------------------------------------------------------

def test_empty_file_rejected():
    response = client.post(
        DETECT_URL,
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 7: Missing file field returns FastAPI's HTTP 422
# ---------------------------------------------------------------------------

def test_missing_file_returns_422():
    response = client.post(DETECT_URL)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 8: Response always includes the four required top-level keys
# ---------------------------------------------------------------------------

def test_response_has_required_keys():
    response = client.post(
        DETECT_URL,
        files={"file": ("frame.png", io.BytesIO(_make_minimal_png()), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    for key in ("status", "model", "image_filename", "detections"):
        assert key in body, f"Missing key in response: {key}"
