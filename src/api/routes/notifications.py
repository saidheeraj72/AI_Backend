from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status

from src.core.config import get_settings
from src.services.socket_manager import manager
from src.api.dependencies.auth import _decode_supabase_token, _build_user_from_claims

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

@router.websocket("/ws/notifications")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time notifications.
    Clients must provide a valid Supabase JWT token as a query parameter.
    """
    user_id: Optional[str] = None
    
    # Authenticate
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        claims = _decode_supabase_token(token, secret=settings.supabase_jwt_secret)
        user = _build_user_from_claims(claims)
        user_id = user.id
    except Exception as e:
        logger.warning("WebSocket authentication failed: %s", e)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Connect
    await manager.connect(user_id, websocket)
    
    try:
        while True:
            # Keep the connection open and listen for messages (heartbeat/echo)
            # We don't expect much upstream traffic from client, but we must await receive
            # to keep the socket alive and detect disconnects.
            data = await websocket.receive_text()
            # Optional: handle client messages if needed (e.g. "ping")
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error("WebSocket error for user %s: %s", user_id, e)
        manager.disconnect(user_id)
