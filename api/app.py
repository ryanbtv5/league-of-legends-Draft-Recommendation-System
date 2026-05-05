"""
api/app.py
-----------
FastAPI application factory and entry point.

Run the API server:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LoL Draft Recommendation API",
        description=(
            "ML-powered League of Legends draft assistant. "
            "Given the current picks and bans, recommends the best next champion pick "
            "to maximise win probability."
        ),
        version=get("project.version", "0.1.0"),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/", tags=["system"])
    def root() -> dict:
        return {
            "service": "LoL Draft Recommendation API",
            "version": get("project.version", "0.1.0"),
            "docs": "/docs",
        }

    logger.info("FastAPI app created — visit /docs for interactive API docs")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=get("api.host", "0.0.0.0"),
        port=get("api.port", 8000),
        reload=get("api.reload", False),
        log_level=get("api.log_level", "info"),
    )
