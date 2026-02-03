"""WebSocket connection manager for real-time updates."""

from typing import Set
from fastapi import WebSocket
import asyncio
import json


class ConnectionManager:
    """Manages WebSocket connections for broadcasting progress updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        if not self.active_connections:
            return
        
        # Use UTF-8 characters directly so the UI can display them (avoid \\uXXXX escapes).
        data = json.dumps(message, ensure_ascii=False)
        async with self._lock:
            dead_connections = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(data)
                except Exception:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            self.active_connections -= dead_connections

    async def send_progress(self, task_id: str, progress: float, message: str, status: str = "running"):
        """Send a progress update for a specific task."""
        await self.broadcast({
            "type": "progress",
            "task_id": task_id,
            "progress": progress,
            "message": message,
            "status": status
        })

    async def send_log(self, task_id: str, line: str, level: str = "info"):
        """Send a log line for a specific task."""
        await self.broadcast({
            "type": "log",
            "task_id": task_id,
            "line": line,
            "level": level
        })

    async def send_complete(self, task_id: str, success: bool, result: dict = None):
        """Send task completion notification."""
        await self.broadcast({
            "type": "complete",
            "task_id": task_id,
            "success": success,
            "result": result or {}
        })


# Global connection manager instance
manager = ConnectionManager()
