from pathlib import Path
from typing import List, Set
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Cheel Nazar"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Base directories (resolved relative to backend directory)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    ROOT_DIR: Path = BASE_DIR.parent
    UPLOADS_DIR: Path = ROOT_DIR / "uploads"
    DATA_DIR: Path = ROOT_DIR / "data"

    # Upload configurations
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mov", ".avi", ".mkv"}
    MAX_UPLOAD_SIZE_MB: int = 500
    MAX_UPLOAD_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB

    # Reconstruction (Phase 8.6).
    # True = the normal processing pipeline EXECUTES reconstruction for the
    # current job at its reconstruction stage (Step 6 of jobs.py), so every
    # newly processed upload ends with real artifacts under that job's data
    # directory. Existing artifacts are always reused and never recomputed;
    # status GET endpoints never trigger reconstruction (they only read disk).
    RECONSTRUCTION_AUTORUN: bool = True

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]

settings = Settings()
