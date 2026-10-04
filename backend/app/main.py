from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.router import api_router
from app.api.v1.endpoints.health import check_health, HealthResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Cheel Nazar: Aerial Reconnaissance & 3D Geospatial Intelligence Engine",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Direct /health endpoint as required by project specification
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def root_health():
    """Root health check endpoint."""
    return await check_health()

# Include versioned API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/", tags=["Root"])
async def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }
