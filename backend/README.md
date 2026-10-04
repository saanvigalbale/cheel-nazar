# Cheel Nazar Backend

FastAPI-powered backend engine for Cheel Nazar drone video ingestion, 3D reconstruction coordination, and AI hazard/damage inference.

---

## Quickstart

### 1. Create Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  # (or .venv\bin\activate if in MSYS2/bash)
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Health Check
Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive OpenAPI documentation.

---

## API Documentation

### Video Upload Endpoint

- **Endpoint**: `POST /api/v1/video/upload`
- **Content-Type**: `multipart/form-data`
- **Supported Formats**: `.mp4`, `.mov`, `.avi`, `.mkv`
- **Max Upload Size**: 500 MB
- **Storage Location**: `uploads/` with sanitized filename prefixed by a generated UUID `job_id`.

#### Example Request (cURL)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/video/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@drone_flight_pass.mp4"
```

#### Example Request (PowerShell)
```powershell
$form = @{
    file = Get-Item -Path ".\drone_flight_pass.mp4"
}
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/video/upload" -Method Post -Form $form
```

#### Example Response (HTTP 201 Created)
```json
{
  "job_id": "8f8b6f37-124b-4c07-b248-cb3a90c58e5e",
  "filename": "8f8b6f37-124b-4c07-b248-cb3a90c58e5e_drone_flight_pass.mp4",
  "status": "uploaded",
  "message": "Video uploaded successfully"
}
```

#### Error Responses
- **HTTP 400 Bad Request**: Invalid/unsupported extension (e.g., `.txt`, `.jpg`, `.exe`) or empty file (0 bytes).
  ```json
  {
    "detail": "Unsupported file format '.txt'. Allowed formats: .avi, .mkv, .mov, .mp4"
  }
  ```
- **HTTP 413 Payload Too Large**: File exceeds maximum allowed upload size (500 MB).
  ```json
  {
    "detail": "File exceeds maximum allowed upload size of 500 MB."
  }
  ```

---

### List Uploaded Videos
- **Endpoint**: `GET /api/v1/video/`
- **Response**:
  ```json
  {
    "total": 1,
    "videos": [
      {
        "job_id": "8f8b6f37-124b-4c07-b248-cb3a90c58e5e",
        "filename": "8f8b6f37-124b-4c07-b248-cb3a90c58e5e_drone_flight_pass.mp4",
        "size_bytes": 10485760,
        "modified_time": 1728038400.0
      }
    ]
  }
  ```

---

## Phase 5 — AI Detection Architecture

### Overview

Phase 5 introduces a **modular AI detection pipeline** with a clean interface
so the `MockDetector` can be swapped for a real YOLO model (Phase 6) without
changing the API or any callers.

```
Uploaded image
     ↓
POST /api/v1/ai/detect
     ↓
Validation (extension, size, empty-file check)
     ↓
BaseDetector.detect(image_path) → List[Detection]
     ↓
MockDetector  ──(Phase 6)──►  YOLODetector
     ↓
Structured JSON response
```

### Architecture

| File | Role |
|---|---|
| `app/services/ai/detector.py` | `BaseDetector` abstract interface, `BoundingBox`, `Detection` dataclasses, `MockDetector` placeholder, module-level `detector` singleton |
| `app/api/v1/endpoints/ai.py` | FastAPI endpoint — decoupled from detector implementation |
| `app/api/v1/router.py` | Registers AI router at `/api/v1/ai` |
| `tests/test_ai_detection.py` | 8 pytest tests — no model weights required |

### Current Implementation

> ⚠ **Phase 5 uses `MockDetector`** — a clearly-marked placeholder that
> returns deterministic dummy data. **No YOLO weights are downloaded or loaded.**
> Real inference will be integrated in Phase 6 via `YOLODetector(BaseDetector)`.

### AI Detection Endpoint

- **Endpoint**: `POST /api/v1/ai/detect`
- **Content-Type**: `multipart/form-data`
- **Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- **Max Image Size**: 50 MB

#### Example Request (cURL)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ai/detect" \
  -H "accept: application/json" \
  -F "file=@frame_000001.jpg"
```

#### Example Request (PowerShell)
```powershell
$form = @{ file = Get-Item -Path ".\frame_000001.jpg" }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/ai/detect" -Method Post -Form $form
```

#### Example Response — MockDetector (HTTP 200)
```json
{
  "status": "success",
  "model": "MockDetector-v0 (placeholder)",
  "image_filename": "frame_000001.jpg",
  "detections": [
    {
      "class_name": "mock_vehicle",
      "confidence": 0.87,
      "bbox": { "x1": 10.0, "y1": 20.0, "x2": 80.0, "y2": 60.0 },
      "is_mock": true
    }
  ]
}
```

> `"is_mock": true` on every detection indicates fabricated data from the placeholder.

#### Error Responses

| HTTP | Cause |
|------|-------|
| 400 | Missing/empty filename, unsupported extension, empty file body |
| 413 | Image exceeds 50 MB |
| 422 | `file` field missing entirely (FastAPI validation) |
| 500 | Unexpected detector failure |

### Running Tests
```bash
# from the backend directory
pip install pytest httpx
python -m pytest tests/ -v
```

### Phase 6 Integration (future)
Replace the `detector` singleton in `app/services/ai/detector.py`:
```python
# Phase 6 — swap one line, nothing else changes
from app.services.ai.yolo_detector import YOLODetector
detector: BaseDetector = YOLODetector(model_path="yolov8n.pt")
```
