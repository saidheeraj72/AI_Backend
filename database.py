from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    mongodb.client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
    mongodb.database = mongodb.client[os.getenv("MONGODB_DATABASE")]
    
    # Create indexes for better performance
    await create_indexes()
    print("Connected to MongoDB")

async def close_mongo_connection():
    """Close database connection"""
    mongodb.client.close()
    print("Disconnected from MongoDB")

async def create_indexes():
    """Create database indexes"""
    # Chat sessions indexes
    await mongodb.database.chat_sessions.create_indexes([
        IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
    ])
    
    # Chat messages indexes
    await mongodb.database.chat_messages.create_indexes([
        IndexModel([("session_id", ASCENDING), ("message_order", ASCENDING)]),
        IndexModel([("session_id", ASCENDING), ("created_at", ASCENDING)]),
    ])

async def get_database():
    """Get database instance"""
    return mongodb.database

# Database Models
class ChatSession:
    @staticmethod
    def create_session(user_id: str, title: str, llm_type: str) -> Dict[str, Any]:
        return {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": title,
            "llm_type": llm_type,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

class ChatMessage:
    @staticmethod
    def create_message(session_id: str, role: str, content: str, message_order: int, image_url: Optional[str] = None) -> Dict[str, Any]:
        return {
            "_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "message_order": message_order,
            "image_url": image_url,
            "created_at": datetime.utcnow()
        }
