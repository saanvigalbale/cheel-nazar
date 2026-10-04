# Cheel Nazar Backend

FastAPI-powered backend engine for Cheel Nazar drone video ingestion, 3D reconstruction coordination, and AI hazard/damage inference.

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
