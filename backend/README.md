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
