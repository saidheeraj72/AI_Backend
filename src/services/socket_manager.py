from __future__ import annotations

import logging
from typing import Dict

from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info("WebSocket connection OPENED for user_id: %s", user_id)

    def disconnect(self, user_id: str) -> None:
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info("WebSocket connection CLOSED for user_id: %s", user_id)

    async def send_personal_message(self, message: str, user_id: str) -> None:
        websocket = self.active_connections.get(user_id)
        if websocket:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error("Failed to send message to user %s: %s", user_id, e)
                # Ideally cleanup if broken
                self.disconnect(user_id)

    async def broadcast(self, message: str) -> None:
        for user_id, connection in list(self.active_connections.items()):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(user_id)

manager = ConnectionManager()
