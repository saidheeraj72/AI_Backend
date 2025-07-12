from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import os
from groq import Groq
from dotenv import load_dotenv
import json
import base64
from typing import Optional, List
import uuid
from datetime import datetime
from database import connect_to_mongo, close_mongo_connection, get_database, ChatSession, ChatMessage

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# LLM Model configurations with vision support
LLM_MODELS = {
    'daily': {
        'model': 'gemma2-9b-it',
        'name': 'Daily Use LLM',
        'system_prompt': 'You are a helpful assistant for everyday tasks. Provide clear, concise answers.',
        'supports_vision': False
    },
    'image': {
        'model': 'llama-3.2-11b-vision-preview',
        'name': 'Image Reasoning LLM',
        'system_prompt': 'You are an expert in image analysis and visual reasoning. Provide detailed insights about images and visual content.',
        'supports_vision': True
    },
    'complex': {
        'model': 'qwen-qwq-32b',
        'name': 'Complex Tasks LLM',
        'system_prompt': 'You are an advanced AI assistant capable of handling complex reasoning, analysis, and problem-solving tasks. Provide thorough, well-reasoned responses.',
        'supports_vision': False
    }
}

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

async def save_chat_session(user_id: str, title: str, llm_type: str) -> str:
    """Create a new chat session"""
    try:
        db = await get_database()
        session_data = ChatSession.create_session(user_id, title, llm_type)
        result = await db.chat_sessions.insert_one(session_data)
        return session_data["_id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")

async def save_message(session_id: str, role: str, content: str, message_order: int, image_url: Optional[str] = None):
    """Save a message to the database"""
    try:
        db = await get_database()
        message_data = ChatMessage.create_message(session_id, role, content, message_order, image_url)
        await db.chat_messages.insert_one(message_data)
        
        # Update session's updated_at timestamp
        await db.chat_sessions.update_one(
            {"_id": session_id},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

async def get_chat_history(session_id: str) -> List[dict]:
    """Retrieve chat history for a session"""
    try:
        db = await get_database()
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("message_order", 1).to_list(None)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@app.post('/chat')
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get('message', '')
    llm_type = data.get('llm_type', 'daily')
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    image_data = data.get('image')  # Base64 encoded image
    
    if not user_message:
        raise HTTPException(status_code=400, detail='`message` is required')
    if llm_type not in LLM_MODELS:
        raise HTTPException(status_code=400, detail='Invalid `llm_type`')

    config = LLM_MODELS[llm_type]
    
    # Create new session if not provided
    if not session_id and user_id:
        # Generate title from first message (first 50 chars)
        title = user_message[:50] + "..." if len(user_message) > 50 else user_message
        session_id = await save_chat_session(user_id, title, llm_type)
    
    # Get existing chat history
    chat_history = []
    if session_id:
        chat_history = await get_chat_history(session_id)
    
    # Build messages for Groq API
    messages = [{'role': 'system', 'content': config['system_prompt']}]
    
    # Add chat history (last 10 messages)
    for msg in chat_history[-10:]:
        if msg['role'] != 'system':
            message_content = {'type': 'text', 'text': msg['content']}
            if msg.get('image_url') and config['supports_vision']:
                messages.append({
                    'role': msg['role'],
                    'content': [
                        message_content,
                        {'type': 'image_url', 'image_url': {'url': msg['image_url']}}
                    ]
                })
            else:
                messages.append({'role': msg['role'], 'content': msg['content']})
    
    # Add current user message
    user_content = [{'type': 'text', 'text': user_message}]
    if image_data and config['supports_vision']:
        user_content.append({
            'type': 'image_url',
            'image_url': {'url': f"data:image/jpeg;base64,{image_data}"}
        })
        
    if config['supports_vision'] and (image_data or any(msg.get('image_url') for msg in chat_history[-5:])):
        messages.append({'role': 'user', 'content': user_content})
    else:
        messages.append({'role': 'user', 'content': user_message})

    try:
        completion = groq_client.chat.completions.create(
            messages=messages,
            model=config['model'],
            temperature=0.7,
            max_tokens=1024,
        )
        response_text = completion.choices[0].message.content
        
        # Save messages to database
        if session_id:
            next_order = len(chat_history)
            await save_message(
                session_id, 
                'user', 
                user_message, 
                next_order,
                f"data:image/jpeg;base64,{image_data}" if image_data else None
            )
            await save_message(session_id, 'assistant', response_text, next_order + 1)
        
        return JSONResponse({
            'response': response_text,
            'llm_type': llm_type,
            'model_name': config['name'],
            'session_id': session_id,
            'status': 'success'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/stream')
async def chat_stream(request: Request):
    data = await request.json()
    user_message = data.get('message', '')
    llm_type = data.get('llm_type', 'daily')
    session_id = data.get('session_id')
    user_id = data.get('user_id')

    if not user_message:
        raise HTTPException(status_code=400, detail='`message` is required')
    if llm_type not in LLM_MODELS:
        raise HTTPException(status_code=400, detail='Invalid `llm_type`')

    config = LLM_MODELS[llm_type]
    
    # Get chat history
    chat_history = []
    if session_id:
        chat_history = await get_chat_history(session_id)
    
    messages = [{'role': 'system', 'content': config['system_prompt']}]
    for msg in chat_history[-10:]:
        if msg['role'] != 'system':
            messages.append({'role': msg['role'], 'content': msg['content']})
    messages.append({'role': 'user', 'content': user_message})

    async def event_generator():
        try:
            stream = groq_client.chat.completions.create(
                messages=messages,
                model=config['model'],
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            
            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'content': delta})}\n\n"
            
            # Save messages after streaming is complete
            if session_id:
                next_order = len(chat_history)
                await save_message(session_id, 'user', user_message, next_order)
                await save_message(session_id, 'assistant', full_response, next_order + 1)
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

@app.post('/upload-image')
async def upload_image(file: UploadFile = File(...)):
    """Upload and convert image to base64"""
    try:
        contents = await file.read()
        base64_encoded = base64.b64encode(contents).decode('utf-8')
        return JSONResponse({
            'image_data': base64_encoded,
            'filename': file.filename,
            'content_type': file.content_type
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.get('/chat-sessions/{user_id}')
async def get_user_chat_sessions(user_id: str):
    """Get all chat sessions for a user"""
    try:
        db = await get_database()
        sessions = await db.chat_sessions.find(
            {"user_id": user_id}
        ).sort("updated_at", -1).to_list(None)
        
        # Convert MongoDB ObjectId to string for JSON serialization
        for session in sessions:
            session["id"] = session["_id"]
            session["created_at"] = session["created_at"].isoformat()
            session["updated_at"] = session["updated_at"].isoformat()
            del session["_id"]
        
        return JSONResponse({'sessions': sessions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/chat-session/{session_id}/messages')
async def get_session_messages(session_id: str):
    """Get all messages for a chat session"""
    try:
        messages = await get_chat_history(session_id)
        
        # Convert MongoDB ObjectId to string for JSON serialization
        for message in messages:
            message["id"] = message["_id"]
            message["created_at"] = message["created_at"].isoformat()
            del message["_id"]
        
        return JSONResponse({'messages': messages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/chat-session/{session_id}')
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages"""
    try:
        db = await get_database()
        # Delete messages first
        await db.chat_messages.delete_many({"session_id": session_id})
        # Delete session
        await db.chat_sessions.delete_one({"_id": session_id})
        return JSONResponse({'status': 'success', 'message': 'Chat session deleted'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/models')
async def get_models():
    return JSONResponse({'models': [
        {
            'id': k, 
            'name': v['name'], 
            'model': v['model'],
            'supports_vision': v['supports_vision']
        } for k, v in LLM_MODELS.items()
    ]})

@app.get('/health')
async def health_check():
    return JSONResponse({'status': 'healthy'})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
