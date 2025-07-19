from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.requests import Request  # Explicit import for FastAPI Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
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
import pickle
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel, EmailStr

# Document processing imports
import numpy as np
import PyPDF2
from docx import Document as DocxDocument

# Vector database and embeddings imports
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Database imports
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

# API imports
from groq import Groq
from dotenv import load_dotenv

# Supabase imports
from supabase import create_client, Client

# Gmail API imports (with aliases to avoid conflicts)
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleRequest  # Aliased import
from google.oauth2.credentials import Credentials

# Load environment variables
load_dotenv()

# Global variables
mongodb_client = None
database = None
doc_processor = None

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Google AI and Pinecone
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Gmail API Configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

# ===== LIFESPAN EVENT HANDLER =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    global doc_processor
    doc_processor = DocumentProcessor()
    doc_processor.load_metadata("./data/metadata")
    
    yield
    
    # Shutdown
    await close_mongo_connection()
    os.makedirs("./data", exist_ok=True)
    if doc_processor:
        doc_processor.save_metadata("./data/metadata")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="AI Chat Backend with RAG and Email Automation", 
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PYDANTIC MODELS =====

# Employee models
class EmployeeCreate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    position: Optional[str] = None
    department: Optional[str] = None

class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    position: Optional[str] = None
    department: Optional[str] = None
    status: Optional[str] = None

class EmployeeResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str]
    position: Optional[str]
    department: Optional[str]
    status: str
    created_at: str
    updated_at: Optional[str]

# Email models
class EmailTemplate(BaseModel):
    name: str
    subject: str
    body: str
    is_html: bool = True

class EmailTemplateUpdate(BaseModel):
    name: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    is_html: Optional[bool] = None

class EmailCampaign(BaseModel):
    name: str
    template_id: int
    recipient_list: List[str]
    scheduled_time: Optional[str] = None

class EmailSend(BaseModel):
    to: List[str]
    subject: str
    body: str
    is_html: bool = True
    template_id: Optional[int] = None

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

# ===== GMAIL API FUNCTIONS =====

def authenticate_gmail():
    """Authenticate and return Gmail service"""
    creds = None
    
    # Token file stores the user's access and refresh tokens
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleRequest())  # Using the aliased GoogleRequest
            except Exception as e:
                print(f"Error refreshing token: {e}")
                creds = None
        
        if not creds:
            if not os.path.exists(CREDENTIALS_FILE):
                raise Exception("credentials.json file not found. Please download it from Google Cloud Console.")
            
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

def create_message(to, subject, body, is_html=True):
    """Create email message"""
    if is_html:
        message = MIMEMultipart('alternative')
        message['to'] = ', '.join(to) if isinstance(to, list) else to
        message['subject'] = subject
        
        # Create both plain text and HTML versions
        text_part = MIMEText(body, 'plain')
        html_part = MIMEText(body, 'html')
        message.attach(text_part)
        message.attach(html_part)
    else:
        message = MIMEText(body, 'plain')
        message['to'] = ', '.join(to) if isinstance(to, list) else to
        message['subject'] = subject
    
    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

def send_email_gmail(to, subject, body, is_html=True):
    """Send email using Gmail API"""
    try:
        service = authenticate_gmail()
        message = create_message(to, subject, body, is_html)
        sent_message = service.users().messages().send(userId='me', body=message).execute()
        return sent_message
    except Exception as e:
        raise Exception(f"Error sending email: {str(e)}")

# ===== DOCUMENT PROCESSOR CLASS =====
class DocumentProcessor:
    def __init__(self):
        self.embedding_model = "text-embedding-004"
        self.dimension = 768  # Google's text-embedding-004 dimension
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ai-chat-rag")
        self.namespace = "documents"
        
        # Initialize Pinecone index
        try:
            # Check if index exists, if not create it
            existing_indexes = [index.name for index in pinecone_client.list_indexes()]
            if self.index_name not in existing_indexes:
                pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            self.index = pinecone_client.Index(self.index_name)
        except Exception as e:
            print(f"Warning: Could not initialize Pinecone index: {e}")
            self.index = None
        
        self.documents = []
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google's embedding model"""
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=f"models/{self.embedding_model}",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            result = genai.embed_content(
                model=f"models/{self.embedding_model}",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Error generating query embedding: {str(e)}")
        
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
            if not self.index:
                raise Exception("Pinecone index not initialized")
                
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
            embeddings = self.get_embeddings(chunks)
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{doc_id}_{i}"
                metadata = {
                    'doc_id': doc_id,
                    'chunk_idx': i,
                    'text': chunk,
                    'filename': filename,
                    'file_type': file_type
                }
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert vectors to Pinecone
            self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
            
            # Store document metadata locally
            self.documents.append(document_metadata)
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.index:
                return []
            
            # Generate query embedding
            query_embedding = self.get_query_embedding(query)
            
            # Search Pinecone
            search_response = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                namespace=self.namespace
            )
            
            results = []
            for match in search_response.matches:
                chunk_data = {
                    'doc_id': match.metadata['doc_id'],
                    'chunk_idx': match.metadata['chunk_idx'],
                    'text': match.metadata['text'],
                    'filename': match.metadata['filename'],
                    'similarity_score': float(match.score)
                }
                results.append(chunk_data)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")
    
    def get_documents(self) -> List[Dict[str, Any]]:
        return self.documents
    
    def delete_document(self, doc_id: str) -> bool:
        try:
            if not self.index:
                return False
                
            # Get all vectors for this document
            query_response = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=10000,  # Large number to get all
                include_metadata=True,
                filter={'doc_id': doc_id},
                namespace=self.namespace
            )
            
            # Delete vectors from Pinecone
            vector_ids = [match.id for match in query_response.matches]
            if vector_ids:
                self.index.delete(ids=vector_ids, namespace=self.namespace)
            
            # Remove from local documents list
            self.documents = [doc for doc in self.documents if doc['id'] != doc_id]
            
            return True
            
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")
    
    def save_metadata(self, filepath: str):
        """Save document metadata to local file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            metadata = {'documents': self.documents}
            with open(f"{filepath}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            raise Exception(f"Error saving metadata: {str(e)}")
    
    def load_metadata(self, filepath: str):
        """Load document metadata from local file"""
        try:
            if os.path.exists(f"{filepath}.json"):
                with open(f"{filepath}.json", 'r') as f:
                    metadata = json.load(f)
                    self.documents = metadata.get('documents', [])
        except Exception as e:
            raise Exception(f"Error loading metadata: {str(e)}")

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

# ===== EMAIL AUTOMATION ENDPOINTS =====

@app.post('/gmail-auth')
async def initialize_gmail_auth():
    """Initialize Gmail authentication"""
    try:
        # This will trigger the OAuth flow
        service = authenticate_gmail()
        return JSONResponse({
            'status': 'success',
            'message': 'Gmail authentication successful'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gmail authentication failed: {str(e)}")

@app.post('/email-templates')
async def create_email_template(template: EmailTemplate):
    """Create a new email template"""
    try:
        template_data = {
            "name": template.name,
            "subject": template.subject,
            "body": template.body,
            "is_html": template.is_html,
            "status": "Active"
        }
        
        result = supabase.table('email_templates').insert(template_data).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'message': 'Email template created successfully',
                'template': result.data[0]
            })
        else:
            raise HTTPException(status_code=400, detail='Failed to create template')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating template: {str(e)}")

@app.get('/email-templates')
async def get_email_templates():
    """Get all email templates"""
    try:
        result = supabase.table('email_templates').select("*").order('created_at', desc=True).execute()
        
        return JSONResponse({
            'status': 'success',
            'templates': result.data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching templates: {str(e)}")

@app.get('/email-templates/{template_id}')
async def get_email_template(template_id: int):
    """Get a specific email template"""
    try:
        result = supabase.table('email_templates').select("*").eq('id', template_id).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'template': result.data[0]
            })
        else:
            raise HTTPException(status_code=404, detail='Template not found')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")

@app.put('/email-templates/{template_id}')
async def update_email_template(template_id: int, template: EmailTemplateUpdate):
    """Update an email template"""
    try:
        update_data = {}
        if template.name is not None:
            update_data['name'] = template.name
        if template.subject is not None:
            update_data['subject'] = template.subject
        if template.body is not None:
            update_data['body'] = template.body
        if template.is_html is not None:
            update_data['is_html'] = template.is_html
        
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase.table('email_templates').update(update_data).eq('id', template_id).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'message': 'Template updated successfully',
                'template': result.data[0]
            })
        else:
            raise HTTPException(status_code=404, detail='Template not found')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating template: {str(e)}")

@app.delete('/email-templates/{template_id}')
async def delete_email_template(template_id: int):
    """Delete an email template"""
    try:
        result = supabase.table('email_templates').delete().eq('id', template_id).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'message': 'Template deleted successfully'
            })
        else:
            raise HTTPException(status_code=404, detail='Template not found')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting template: {str(e)}")

@app.post('/send-email')
async def send_single_email(email_data: EmailSend):
    """Send a single email"""
    try:
        # Send email using Gmail API
        sent_message = send_email_gmail(
            to=email_data.to,
            subject=email_data.subject,
            body=email_data.body,
            is_html=email_data.is_html
        )
        
        # Log the email in database
        for recipient in email_data.to:
            log_data = {
                "recipient": recipient,
                "subject": email_data.subject,
                "body": email_data.body,
                "status": "Sent",
                "template_id": email_data.template_id,
                "message_id": sent_message.get('id')
            }
            supabase.table('email_logs').insert(log_data).execute()
        
        return JSONResponse({
            'status': 'success',
            'message': f'Email sent to {len(email_data.to)} recipients',
            'message_id': sent_message.get('id')
        })
        
    except Exception as e:
        # Log failed email
        for recipient in email_data.to:
            log_data = {
                "recipient": recipient,
                "subject": email_data.subject,
                "body": email_data.body,
                "status": "Failed",
                "template_id": email_data.template_id,
                "error_message": str(e)
            }
            supabase.table('email_logs').insert(log_data).execute()
        
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

@app.post('/email-campaigns')
async def create_email_campaign(campaign: EmailCampaign):
    """Create and execute email campaign"""
    try:
        # Get template
        template_result = supabase.table('email_templates').select("*").eq('id', campaign.template_id).execute()
        
        if not template_result.data:
            raise HTTPException(status_code=404, detail='Template not found')
        
        template = template_result.data[0]
        
        # Create campaign record
        campaign_data = {
            "name": campaign.name,
            "template_id": campaign.template_id,
            "total_recipients": len(campaign.recipient_list),
            "status": "In Progress"
        }
        
        campaign_result = supabase.table('email_campaigns').insert(campaign_data).execute()
        campaign_id = campaign_result.data[0]['id']
        
        sent_count = 0
        failed_count = 0
        
        # Send emails to all recipients
        for recipient in campaign.recipient_list:
            try:
                sent_message = send_email_gmail(
                    to=[recipient],
                    subject=template['subject'],
                    body=template['body'],
                    is_html=template['is_html']
                )
                
                # Log successful email
                log_data = {
                    "recipient": recipient,
                    "subject": template['subject'],
                    "body": template['body'],
                    "status": "Sent",
                    "template_id": campaign.template_id,
                    "campaign_id": campaign_id,
                    "message_id": sent_message.get('id')
                }
                supabase.table('email_logs').insert(log_data).execute()
                sent_count += 1
                
            except Exception as e:
                # Log failed email
                log_data = {
                    "recipient": recipient,
                    "subject": template['subject'],
                    "body": template['body'],
                    "status": "Failed",
                    "template_id": campaign.template_id,
                    "campaign_id": campaign_id,
                    "error_message": str(e)
                }
                supabase.table('email_logs').insert(log_data).execute()
                failed_count += 1
        
        # Update campaign with results
        update_data = {
            "sent_count": sent_count,
            "failed_count": failed_count,
            "status": "Completed",
            "completed_at": datetime.utcnow().isoformat()
        }
        supabase.table('email_campaigns').update(update_data).eq('id', campaign_id).execute()
        
        return JSONResponse({
            'status': 'success',
            'message': f'Campaign completed. Sent: {sent_count}, Failed: {failed_count}',
            'campaign_id': campaign_id,
            'sent_count': sent_count,
            'failed_count': failed_count
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating campaign: {str(e)}")

@app.get('/email-campaigns')
async def get_email_campaigns():
    """Get all email campaigns"""
    try:
        result = supabase.table('email_campaigns').select("*").order('created_at', desc=True).execute()
        
        return JSONResponse({
            'status': 'success',
            'campaigns': result.data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching campaigns: {str(e)}")

@app.get('/email-logs')
async def get_email_logs(limit: int = 100):
    """Get email logs"""
    try:
        result = supabase.table('email_logs').select("*").order('sent_at', desc=True).limit(limit).execute()
        
        return JSONResponse({
            'status': 'success',
            'logs': result.data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {str(e)}")

@app.get('/email-analytics')
async def get_email_analytics():
    """Get email analytics"""
    try:
        # Get total counts
        logs_result = supabase.table('email_logs').select("status").execute()
        campaigns_result = supabase.table('email_campaigns').select("*").execute()
        
        logs = logs_result.data
        campaigns = campaigns_result.data
        
        total_sent = sum(1 for log in logs if log['status'] == 'Sent')
        total_failed = sum(1 for log in logs if log['status'] == 'Failed')
        total_delivered = total_sent  # Assuming sent = delivered for now
        total_campaigns = len(campaigns)
        
        # Calculate rates (placeholder - you'd need actual tracking for opens/clicks)
        open_rate = 0.68  # 68%
        click_rate = 0.13  # 13%
        
        estimated_opens = int(total_delivered * open_rate)
        estimated_clicks = int(total_delivered * click_rate)
        
        return JSONResponse({
            'status': 'success',
            'analytics': {
                'total_sent': total_sent + total_failed,
                'total_delivered': total_delivered,
                'total_failed': total_failed,
                'total_opened': estimated_opens,
                'total_clicked': estimated_clicks,
                'total_campaigns': total_campaigns,
                'delivery_rate': round((total_delivered / (total_sent + total_failed) * 100), 2) if (total_sent + total_failed) > 0 else 0,
                'open_rate': round(open_rate * 100, 2),
                'click_rate': round(click_rate * 100, 2)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

# ===== EMPLOYEE ENDPOINTS =====

@app.post('/employees')
async def create_employee(employee: EmployeeCreate):
    """Create a new employee"""
    try:
        employee_data = {
            "name": employee.name,
            "email": employee.email,
            "phone": employee.phone,
            "position": employee.position,
            "department": employee.department,
            "status": "Active"
        }
        
        result = supabase.table('employees').insert(employee_data).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'message': 'Employee created successfully',
                'employee': result.data[0]
            })
        else:
            raise HTTPException(status_code=400, detail='Failed to create employee')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating employee: {str(e)}")

@app.get('/employees')
async def get_employees():
    """Get all employees"""
    try:
        result = supabase.table('employees').select("*").order('created_at', desc=True).execute()
        
        return JSONResponse({
            'status': 'success',
            'employees': result.data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching employees: {str(e)}")

@app.get('/employees/{employee_id}')
async def get_employee(employee_id: int):
    """Get a specific employee"""
    try:
        result = supabase.table('employees').select("*").eq('id', employee_id).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'employee': result.data[0]
            })
        else:
            raise HTTPException(status_code=404, detail='Employee not found')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching employee: {str(e)}")

@app.put('/employees/{employee_id}')
async def update_employee(employee_id: int, employee: EmployeeUpdate):
    """Update an employee"""
    try:
        update_data = {}
        if employee.name is not None:
            update_data['name'] = employee.name
        if employee.email is not None:
            update_data['email'] = employee.email
        if employee.phone is not None:
            update_data['phone'] = employee.phone
        if employee.position is not None:
            update_data['position'] = employee.position
        if employee.department is not None:
            update_data['department'] = employee.department
        if employee.status is not None:
            update_data['status'] = employee.status
        
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase.table('employees').update(update_data).eq('id', employee_id).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'message': 'Employee updated successfully',
                'employee': result.data[0]
            })
        else:
            raise HTTPException(status_code=404, detail='Employee not found')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating employee: {str(e)}")

@app.delete('/employees/{employee_id}')
async def delete_employee(employee_id: int):
    """Delete an employee"""
    try:
        result = supabase.table('employees').delete().eq('id', employee_id).execute()
        
        if result.data:
            return JSONResponse({
                'status': 'success',
                'message': 'Employee deleted successfully'
            })
        else:
            raise HTTPException(status_code=404, detail='Employee not found')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting employee: {str(e)}")

@app.post('/employees/bulk-import')
async def bulk_import_employees(file: UploadFile = File(...)):
    """Import employees from Excel/CSV file"""
    try:
        # This is a placeholder for Excel/CSV import functionality
        # You would implement the actual Excel parsing here using pandas or openpyxl
        return JSONResponse({
            'status': 'success',
            'message': 'Bulk import functionality - implementation needed',
            'filename': file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing employees: {str(e)}")

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
    """RAG-enabled chat endpoint"""
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
        
        # Save metadata
        os.makedirs("./data", exist_ok=True)
        doc_processor.save_metadata("./data/metadata")
        
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
            doc_processor.save_metadata("./data/metadata")
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
        'documents_indexed': len(doc_processor.documents) if doc_processor else 0,
        'pinecone_connected': doc_processor.index is not None if doc_processor else False,
        'gmail_auth_available': os.path.exists(CREDENTIALS_FILE)
    })

# ===== RUN SERVER =====
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
