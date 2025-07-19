from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import os
import uuid
import json
import base64
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from io import BytesIO
import tempfile
import shutil

# Document processing imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document as DocxDocument

# Database imports
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

# API imports
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Chat Backend with RAG", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
mongodb_client = None
database = None

# LLM Model configurations
LLM_MODELS = {
    'daily': {
        'model': 'gemma2-9b-it',
        'name': 'Daily Use LLM',
        'system_prompt': 'You are a helpful assistant for everyday tasks. Provide clear, concise answers.',
        'supports_vision': False
    },
    'image': {
        'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
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

# ===== DOCUMENT PROCESSOR CLASS =====
class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.chunks = []
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        try:
            doc = DocxDocument(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {str(e)}")
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            raise Exception(f"Error extracting TXT text: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def process_document(self, file_content: bytes, filename: str, file_type: str) -> str:
        try:
            # Extract text based on file type
            if file_type == 'application/pdf':
                text = self.extract_text_from_pdf(file_content)
            elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text = self.extract_text_from_docx(file_content)
            elif file_type == 'text/plain':
                text = self.extract_text_from_txt(file_content)
            else:
                raise Exception(f"Unsupported file type: {file_type}")
            
            # Create document metadata
            doc_id = str(uuid.uuid4())
            document_metadata = {
                'id': doc_id,
                'filename': filename,
                'file_type': file_type,
                'processed_at': datetime.utcnow().isoformat(),
                'text_length': len(text)
            }
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Generate embeddings
            embeddings = self.model.encode(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            chunk_start_idx = len(self.chunks)
            for i, chunk in enumerate(chunks):
                self.chunks.append({
                    'doc_id': doc_id,
                    'chunk_idx': chunk_start_idx + i,
                    'text': chunk,
                    'filename': filename
                })
            
            self.documents.append(document_metadata)
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if self.index.ntotal == 0:
                return []
            
            query_embedding = self.model.encode([query]).astype('float32')
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    chunk_data = self.chunks[idx].copy()
                    chunk_data['similarity_score'] = float(distances[0][i])
                    results.append(chunk_data)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")
    
    def get_documents(self) -> List[Dict[str, Any]]:
        return self.documents
    
    def delete_document(self, doc_id: str) -> bool:
        try:
            self.documents = [doc for doc in self.documents if doc['id'] != doc_id]
            self.chunks = [chunk for chunk in self.chunks if chunk['doc_id'] != doc_id]
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            if self.chunks:
                texts = [chunk['text'] for chunk in self.chunks]
                embeddings = self.model.encode(texts)
                self.index.add(embeddings.astype('float32'))
            
            return True
            
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")
    
    def save_index(self, filepath: str):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            faiss.write_index(self.index, f"{filepath}.index")
            
            metadata = {
                'documents': self.documents,
                'chunks': self.chunks
            }
            with open(f"{filepath}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str):
        try:
            if os.path.exists(f"{filepath}.index"):
                self.index = faiss.read_index(f"{filepath}.index")
            
            if os.path.exists(f"{filepath}.json"):
                with open(f"{filepath}.json", 'r') as f:
                    metadata = json.load(f)
                    self.documents = metadata.get('documents', [])
                    self.chunks = metadata.get('chunks', [])
                    
        except Exception as e:
            raise Exception(f"Error loading index: {str(e)}")

# Global document processor instance
doc_processor = DocumentProcessor()

# ===== DATABASE MODELS =====
class ChatSession:
    @staticmethod
    def create_session(user_id: str, title: str, llm_type: str) -> dict:
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
    def create_message(session_id: str, role: str, content: str, message_order: int, image_url: Optional[str] = None) -> dict:
        return {
            "_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "message_order": message_order,
            "image_url": image_url,
            "created_at": datetime.utcnow()
        }

# ===== DATABASE FUNCTIONS =====
async def connect_to_mongo():
    global mongodb_client, database
    mongodb_client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
    database = mongodb_client.ai_chat_db
    
    # Create indexes
    await database.chat_sessions.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
    await database.chat_messages.create_index([("session_id", ASCENDING), ("message_order", ASCENDING)])

async def close_mongo_connection():  
    if mongodb_client:
        mongodb_client.close()
 
async def get_database():
    return database

async def save_chat_session(user_id: str, title: str, llm_type: str) -> str:
    try:
        db = await get_database()
        session_data = ChatSession.create_session(user_id, title, llm_type)
        result = await db.chat_sessions.insert_one(session_data)
        return session_data["_id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")

async def save_message(session_id: str, role: str, content: str, message_order: int, image_url: Optional[str] = None):
    try:
        db = await get_database()
        message_data = ChatMessage.create_message(session_id, role, content, message_order, image_url)
        await db.chat_messages.insert_one(message_data)
        
        # Update session timestamp
        await db.chat_sessions.update_one(
            {"_id": session_id},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

async def get_chat_history(session_id: str) -> List[dict]:
    try:
        db = await get_database()
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("message_order", 1).to_list(None)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

# ===== EVENT HANDLERS =====
@app.on_event("startup")
async def startup_events():
    await connect_to_mongo()
    doc_processor.load_index("./data/faiss_index")

@app.on_event("shutdown")
async def shutdown_events():
    await close_mongo_connection()
    os.makedirs("./data", exist_ok=True)
    doc_processor.save_index("./data/faiss_index")

# ===== CHAT ENDPOINTS =====
@app.post('/chat')
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get('message', '')
    llm_type = data.get('llm_type', 'daily')
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    image_data = data.get('image')
    
    if not user_message:
        raise HTTPException(status_code=400, detail='`message` is required')
    if llm_type not in LLM_MODELS:
        raise HTTPException(status_code=400, detail='Invalid `llm_type`')

    config = LLM_MODELS[llm_type]
    
    # Create new session if not provided
    if not session_id and user_id:
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

@app.post('/chat/rag')
async def rag_chat(request: Request):
    """RAG-enabled chat endpoint - NO SOURCES RETURNED"""
    try:
        data = await request.json()
        user_message = data.get('message', '')
        llm_type = data.get('llm_type', 'daily')
        session_id = data.get('session_id')
        user_id = data.get('user_id')
        selected_documents = data.get('selected_documents', [])
        
        if not user_message:
            raise HTTPException(status_code=400, detail='`message` is required')
        if llm_type not in LLM_MODELS:
            raise HTTPException(status_code=400, detail='Invalid `llm_type`')
        
        config = LLM_MODELS[llm_type]
        
        # Create new session if not provided
        if not session_id and user_id:
            title = f"RAG: {user_message[:50]}..." if len(user_message) > 50 else f"RAG: {user_message}"
            session_id = await save_chat_session(user_id, title, llm_type)
        
        # Get existing chat history
        chat_history = []
        if session_id:
            chat_history = await get_chat_history(session_id)
        
        # Search for relevant documents
        relevant_chunks = doc_processor.search_similar_chunks(user_message, k=5)
        
        # Build context from relevant chunks
        context = ""
        if relevant_chunks:
            context = "\n\n".join([
                f"Document: {chunk['filename']}\nContent: {chunk['text'][:500]}..."
                for chunk in relevant_chunks
            ])
        
        # Build messages for Groq API
        system_prompt = f"""You are a helpful assistant that answers questions based on provided context. 
        Use the following context to answer the user's question. If the context doesn't contain relevant information, 
        say so and provide a general response.

        Context:
        {context}

        Instructions:
        - Answer based primarily on the provided context
        - If context is insufficient, acknowledge this and provide general knowledge
        - Be concise and accurate
        - Do not mention document names or sources in your response
        """
        
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # Add chat history (last 5 messages)
        for msg in chat_history[-5:]:
            if msg['role'] != 'system':
                messages.append({'role': msg['role'], 'content': msg['content']})
        
        # Add current user message
        messages.append({'role': 'user', 'content': user_message})
        
        # Call Groq API
        completion = groq_client.chat.completions.create(
            messages=messages,
            model=config['model'],
            temperature=0.7,
            max_tokens=1024,
        )
        
        response_text = completion.choices[0].message.content
        
        # DO NOT add source information to response
        
        # Save messages to database
        if session_id:
            next_order = len(chat_history)
            await save_message(session_id, 'user', user_message, next_order)
            await save_message(session_id, 'assistant', response_text, next_order + 1)
        
        return JSONResponse({
            'response': response_text,
            'llm_type': llm_type,
            'model_name': config['name'],
            'session_id': session_id,
            'relevant_chunks': len(relevant_chunks),
            'status': 'success'
            # NO SOURCES RETURNED
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
            
            # Save messages after streaming
            if session_id:
                next_order = len(chat_history)
                await save_message(session_id, 'user', user_message, next_order)
                await save_message(session_id, 'assistant', full_response, next_order + 1)
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

# ===== DOCUMENT ENDPOINTS =====
@app.post('/upload-document')
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for RAG"""
    try:
        allowed_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain'
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        file_content = await file.read()
        doc_id = doc_processor.process_document(file_content, file.filename, file.content_type)
        
        # Save index
        os.makedirs("./data", exist_ok=True)
        doc_processor.save_index("./data/faiss_index")
        
        return JSONResponse({
            'doc_id': doc_id,
            'filename': file.filename,
            'status': 'success',
            'message': 'Document processed successfully'
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/documents')
async def get_documents():
    """Get all processed documents"""
    try:
        documents = doc_processor.get_documents()
        return JSONResponse({'documents': documents})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/document/{doc_id}')
async def delete_document(doc_id: str):
    """Delete a processed document"""
    try:
        success = doc_processor.delete_document(doc_id)
        if success:
            os.makedirs("./data", exist_ok=True)
            doc_processor.save_index("./data/faiss_index")
            return JSONResponse({
                'status': 'success',
                'message': 'Document deleted successfully'
            })
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/search-documents')
async def search_documents(request: Request):
    """Search through documents"""
    try:
        data = await request.json()
        query = data.get('query', '')
        k = data.get('k', 5)
        
        if not query:
            raise HTTPException(status_code=400, detail='Query is required')
        
        results = doc_processor.search_similar_chunks(query, k)
        
        return JSONResponse({
            'results': results,
            'query': query,
            'total_results': len(results)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== IMAGE ENDPOINTS =====
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

# ===== SESSION ENDPOINTS =====
@app.get('/chat-sessions/{user_id}')
async def get_user_chat_sessions(user_id: str):
    """Get all chat sessions for a user"""
    try:
        db = await get_database()
        sessions = await db.chat_sessions.find(
            {"user_id": user_id}
        ).sort("updated_at", -1).to_list(None)
        
        # Convert ObjectId to string and format dates
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
        
        # Convert ObjectId to string and format dates
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

# ===== UTILITY ENDPOINTS =====
@app.get('/models')
async def get_models():
    """Get available LLM models"""
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
    """Health check endpoint"""
    return JSONResponse({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'documents_indexed': len(doc_processor.documents),
        'chunks_indexed': len(doc_processor.chunks)
    })

# ===== RUN SERVER =====
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
