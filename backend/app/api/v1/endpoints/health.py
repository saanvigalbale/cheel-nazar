from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    timestamp: str

@router.get("/health", response_model=HealthResponse)
async def check_health() -> HealthResponse:
    """Check health status of the Cheel Nazar backend."""
    return HealthResponse(
        status="ok",
        app="Cheel Nazar Backend",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat()
    )
