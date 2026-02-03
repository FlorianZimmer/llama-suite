"""
llama-suite Web UI Server

FastAPI-based web server providing a modern UI for managing llama-suite.

Run with:
    python -m llama_suite.webui.server

Or:
    uvicorn llama_suite.webui.server:app --reload --port 8088
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add src to path if needed
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from llama_suite.webui import STATIC_DIR  # noqa: E402
from llama_suite.webui.utils.ws_manager import manager as ws_manager  # noqa: E402
from llama_suite.webui.utils.auth import require_api_key, websocket_authenticated  # noqa: E402

# Import API routers
from llama_suite.webui.api.auth import router as auth_router  # noqa: E402
from llama_suite.webui.api.config import router as config_router  # noqa: E402
from llama_suite.webui.api.config_studio import router as config_studio_router  # noqa: E402
from llama_suite.webui.api.models import router as models_router  # noqa: E402
from llama_suite.webui.api.results import router as results_router  # noqa: E402
from llama_suite.webui.api.bench import router as bench_router  # noqa: E402
from llama_suite.webui.api.memory import router as memory_router  # noqa: E402
from llama_suite.webui.api.eval import router as eval_router  # noqa: E402
from llama_suite.webui.api.sweeps import router as sweeps_router  # noqa: E402
from llama_suite.webui.api.watcher import router as watcher_router  # noqa: E402
from llama_suite.webui.api.system import router as system_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("=" * 60)
    print("  llama-suite Web UI")
    print("=" * 60)
    print(f"  Static files: {STATIC_DIR}")
    print("  Visit: http://localhost:8088")
    print("=" * 60)
    yield
    print("\nShutting down llama-suite Web UI...")


# Create FastAPI app
app = FastAPI(
    title="llama-suite Web UI",
    description="Web-based management interface for llama-suite",
    version="0.1.0",
    lifespan=lifespan
)

# Include API routers
app.include_router(auth_router)
app.include_router(config_router, dependencies=[Depends(require_api_key)])
app.include_router(config_studio_router, dependencies=[Depends(require_api_key)])
app.include_router(models_router, dependencies=[Depends(require_api_key)])
app.include_router(results_router, dependencies=[Depends(require_api_key)])
app.include_router(bench_router, dependencies=[Depends(require_api_key)])
app.include_router(memory_router, dependencies=[Depends(require_api_key)])
app.include_router(eval_router, dependencies=[Depends(require_api_key)])
app.include_router(sweeps_router, dependencies=[Depends(require_api_key)])
app.include_router(watcher_router, dependencies=[Depends(require_api_key)])
app.include_router(system_router, dependencies=[Depends(require_api_key)])

@app.middleware("http")
async def _no_cache_static_assets(request, call_next):
    """
    Avoid UI confusion during local development by disabling caching for /static assets.
    """
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# WebSocket endpoint for real-time updates
@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for receiving real-time progress updates."""
    if not websocket_authenticated(websocket):
        await websocket.close(code=4401)
        return
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            data = await websocket.receive_text()
            # Client can send ping messages to keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "llama-suite-webui"}


# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Serve index.html for root and any unmatched routes (SPA support)
@app.get("/")
async def serve_root():
    """Serve the main application."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, headers={"Cache-Control": "no-store"})
    return {"message": "llama-suite Web UI", "note": "Static files not found. Please check installation."}


@app.get("/{path:path}")
async def serve_spa(path: str):
    """Serve static files or fallback to index.html for SPA routing."""
    # First try to serve static file
    static_path = STATIC_DIR / path
    if static_path.exists() and static_path.is_file():
        return FileResponse(static_path, headers={"Cache-Control": "no-store"})
    
    # Fallback to index.html for SPA routing
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, headers={"Cache-Control": "no-store"})
    
    return {"error": "Not found", "path": path}


def main():
    """Run the server."""
    import uvicorn
    
    print("\n" + "=" * 60)
    print("  Starting llama-suite Web UI Server")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "llama_suite.webui.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8088")),
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
