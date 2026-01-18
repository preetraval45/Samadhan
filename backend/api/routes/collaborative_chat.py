"""
Collaborative Chat - Multi-user real-time chat with WebSocket support
Allows multiple users to collaborate in the same conversation
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from typing import Dict, List, Set
import json
import asyncio
from datetime import datetime
import uuid

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        # conversation_id -> set of active connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # websocket -> user_info
        self.user_info: Dict[WebSocket, dict] = {}
        # conversation_id -> list of typing users
        self.typing_users: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, conversation_id: str, user_id: str, username: str):
        await websocket.accept()

        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = set()
            self.typing_users[conversation_id] = set()

        self.active_connections[conversation_id].add(websocket)
        self.user_info[websocket] = {
            'user_id': user_id,
            'username': username,
            'conversation_id': conversation_id,
            'joined_at': datetime.now().isoformat()
        }

        # Notify others that user joined
        await self.broadcast_presence(conversation_id, {
            'type': 'user_joined',
            'user_id': user_id,
            'username': username,
            'timestamp': datetime.now().isoformat()
        })

        # Send list of active users to new connection
        active_users = [
            {'user_id': info['user_id'], 'username': info['username']}
            for ws, info in self.user_info.items()
            if info['conversation_id'] == conversation_id
        ]
        await websocket.send_json({
            'type': 'active_users',
            'users': active_users
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.user_info:
            user_data = self.user_info[websocket]
            conversation_id = user_data['conversation_id']
            user_id = user_data['user_id']
            username = user_data['username']

            # Remove from active connections
            if conversation_id in self.active_connections:
                self.active_connections[conversation_id].discard(websocket)

                if not self.active_connections[conversation_id]:
                    del self.active_connections[conversation_id]
                    if conversation_id in self.typing_users:
                        del self.typing_users[conversation_id]

            # Remove typing indicator
            if conversation_id in self.typing_users:
                self.typing_users[conversation_id].discard(user_id)

            del self.user_info[websocket]

            # Notify others that user left
            asyncio.create_task(
                self.broadcast_presence(conversation_id, {
                    'type': 'user_left',
                    'user_id': user_id,
                    'username': username,
                    'timestamp': datetime.now().isoformat()
                })
            )

    async def broadcast_message(self, conversation_id: str, message: dict, exclude: WebSocket = None):
        if conversation_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[conversation_id]:
                if connection != exclude:
                    try:
                        await connection.send_json(message)
                    except:
                        disconnected.append(connection)

            # Clean up disconnected connections
            for ws in disconnected:
                self.disconnect(ws)

    async def broadcast_presence(self, conversation_id: str, message: dict):
        await self.broadcast_message(conversation_id, message)

    async def broadcast_typing(self, conversation_id: str, user_id: str, username: str, is_typing: bool):
        if conversation_id in self.typing_users:
            if is_typing:
                self.typing_users[conversation_id].add(user_id)
            else:
                self.typing_users[conversation_id].discard(user_id)

            await self.broadcast_message(conversation_id, {
                'type': 'typing',
                'user_id': user_id,
                'username': username,
                'is_typing': is_typing,
                'typing_users': list(self.typing_users[conversation_id])
            })

    async def broadcast_cursor(self, conversation_id: str, user_id: str, username: str, position: dict):
        await self.broadcast_message(conversation_id, {
            'type': 'cursor_position',
            'user_id': user_id,
            'username': username,
            'position': position
        })


manager = ConnectionManager()


@router.websocket("/ws/collaborative/{conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    conversation_id: str,
    user_id: str,
    username: str
):
    await manager.connect(websocket, conversation_id, user_id, username)

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            msg_type = message_data.get('type')

            if msg_type == 'message':
                # Broadcast chat message to all participants
                await manager.broadcast_message(
                    conversation_id,
                    {
                        'type': 'message',
                        'message_id': str(uuid.uuid4()),
                        'user_id': user_id,
                        'username': username,
                        'content': message_data.get('content'),
                        'timestamp': datetime.now().isoformat(),
                        'role': 'user'
                    },
                    exclude=websocket
                )

            elif msg_type == 'typing':
                # Broadcast typing indicator
                await manager.broadcast_typing(
                    conversation_id,
                    user_id,
                    username,
                    message_data.get('is_typing', False)
                )

            elif msg_type == 'cursor':
                # Broadcast cursor position for collaborative editing
                await manager.broadcast_cursor(
                    conversation_id,
                    user_id,
                    username,
                    message_data.get('position', {})
                )

            elif msg_type == 'ai_response':
                # Broadcast AI response to all participants
                await manager.broadcast_message(
                    conversation_id,
                    {
                        'type': 'message',
                        'message_id': str(uuid.uuid4()),
                        'user_id': 'assistant',
                        'username': 'AI Assistant',
                        'content': message_data.get('content'),
                        'timestamp': datetime.now().isoformat(),
                        'role': 'assistant',
                        'model': message_data.get('model'),
                        'sources': message_data.get('sources', [])
                    }
                )

            elif msg_type == 'reaction':
                # Broadcast message reaction (like, emoji, etc.)
                await manager.broadcast_message(
                    conversation_id,
                    {
                        'type': 'reaction',
                        'message_id': message_data.get('message_id'),
                        'user_id': user_id,
                        'username': username,
                        'reaction': message_data.get('reaction'),
                        'timestamp': datetime.now().isoformat()
                    },
                    exclude=websocket
                )

            elif msg_type == 'highlight':
                # Broadcast text highlight for collaborative annotation
                await manager.broadcast_message(
                    conversation_id,
                    {
                        'type': 'highlight',
                        'user_id': user_id,
                        'username': username,
                        'message_id': message_data.get('message_id'),
                        'range': message_data.get('range'),
                        'color': message_data.get('color', 'yellow')
                    },
                    exclude=websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.post("/collaborative/conversations")
async def create_collaborative_conversation(
    title: str,
    creator_id: str,
    creator_username: str
):
    """Create a new collaborative conversation"""
    conversation_id = str(uuid.uuid4())

    # In production, save to database
    conversation = {
        'conversation_id': conversation_id,
        'title': title,
        'creator_id': creator_id,
        'creator_username': creator_username,
        'created_at': datetime.now().isoformat(),
        'participants': [creator_id],
        'is_collaborative': True
    }

    return conversation


@router.post("/collaborative/conversations/{conversation_id}/invite")
async def invite_to_conversation(
    conversation_id: str,
    user_id: str,
    invited_by: str
):
    """Invite a user to join a collaborative conversation"""

    # In production, save invitation to database and send notification
    invitation = {
        'conversation_id': conversation_id,
        'user_id': user_id,
        'invited_by': invited_by,
        'invited_at': datetime.now().isoformat(),
        'status': 'pending'
    }

    return invitation


@router.get("/collaborative/conversations/{conversation_id}/participants")
async def get_participants(conversation_id: str):
    """Get list of participants in a collaborative conversation"""

    active_users = [
        {
            'user_id': info['user_id'],
            'username': info['username'],
            'joined_at': info['joined_at'],
            'is_online': True
        }
        for ws, info in manager.user_info.items()
        if info['conversation_id'] == conversation_id
    ]

    return {
        'conversation_id': conversation_id,
        'active_participants': active_users,
        'total_online': len(active_users)
    }


@router.get("/collaborative/conversations/{conversation_id}/status")
async def get_conversation_status(conversation_id: str):
    """Get real-time status of a collaborative conversation"""

    typing_users = list(manager.typing_users.get(conversation_id, set()))
    active_count = len(manager.active_connections.get(conversation_id, set()))

    return {
        'conversation_id': conversation_id,
        'active_users': active_count,
        'typing_users': typing_users,
        'is_active': active_count > 0
    }
