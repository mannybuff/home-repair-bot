# app/main.py

"""
Main FastAPI application factory.
Wires up all v2 routers cleanly.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from .core.settings import settings
from app.security.auth import require_api_key, ratelimit_dependency

# --- API Routers (v2 names only) ---

from app.api.image_routes import router as image_router
from app.api.session_routes import router_sessions, router_dialog
from app.api.safety_routes import router as safety_router
from app.api.orchestrator import router as orchestrator_router


def create_app() -> FastAPI:
    app = FastAPI(title="Vision-RAG API", version="v2")

    # allow reverse proxies
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

    # CORS (development wide-open, to tighten later)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Global dependencies ---
    if settings.api_keys:
        app.router.dependencies.append(Depends(require_api_key))

    app.router.dependencies.append(Depends(ratelimit_dependency))

    # --- Mount routers ---

    app.include_router(image_router)
    app.include_router(router_sessions)
    app.include_router(router_dialog)
    app.include_router(safety_router)
    app.include_router(orchestrator_router)

    return app


app = create_app()
