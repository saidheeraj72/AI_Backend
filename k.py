import streamlit as st
import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import camelot
from typing import List, Dict, Tuple, Optional
import re
import numpy as np
from datetime import datetime
import pickle
import json
import logging
from logging.handlers import RotatingFileHandler
import traceback
import time
from dotenv import load_dotenv
from langchain.schema.embeddings import Embeddings

# Vector store and embeddings
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
import warnings
from langchain_core._api import LangChainDeprecationWarning
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('RAG_Chatbot')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / f'rag_chatbot_{datetime.now().strftime("%Y%m%d")}.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Import your PDF processor with logging
class PDFProcessor:
    """
    Comprehensive PDF processor for converting to HTML and extracting text/tables
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.output_dir = Path(pdf_path).parent / f"{Path(pdf_path).stem}_output"
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"PDFProcessor initialized for: {pdf_path}")
        
    def extract_text_pymupdf(self) -> List[Dict]:
        """Extract text using PyMuPDF with formatting preservation"""
        logger.info(f"Extracting text from PDF using PyMuPDF")
        try:
            doc = fitz.open(self.pdf_path)
            pages_data = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text with formatting
                text_dict = page.get_text("dict")
                plain_text = page.get_text()
                
                # Extract images info
                image_list = page.get_images()
                
                pages_data.append({
                    'page_num': page_num + 1,
                    'text': plain_text,
                    'text_dict': text_dict,
                    'images': len(image_list)
                })
                logger.debug(f"Extracted page {page_num + 1}: {len(plain_text)} chars, {len(image_list)} images")
                
            doc.close()
            logger.info(f"Successfully extracted text from {len(pages_data)} pages")
            return pages_data
            
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {str(e)}", exc_info=True)
            raise
    
    def extract_tables_camelot(self) -> List[pd.DataFrame]:
        """Extract tables using Camelot (more accurate for complex tables)"""
        logger.info("Extracting tables using Camelot")
        try:
            # Extract tables from all pages
            tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='lattice')
            
            if not tables:
                logger.debug("No tables found with lattice flavor, trying stream flavor")
                # Try stream flavor if lattice fails
                tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='stream')
            
            logger.info(f"Camelot extracted {len(tables)} tables")
            return [table.df for table in tables]
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
            return []
    
    def extract_tables_pdfplumber(self) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber (good for simple tables)"""
        logger.info("Extracting tables using pdfplumber")
        tables = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:  # Ensure table has data
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
                            logger.debug(f"Found table on page {i+1}: {df.shape}")
            
            logger.info(f"pdfplumber extracted {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return []
    
    def get_combined_content(self) -> Tuple[str, List[Dict], List[pd.DataFrame]]:
        """Get combined text and table content for embedding with better formatting"""
        logger.info("Combining PDF content for processing")
        
        # Extract text
        pages_data = self.extract_text_pymupdf()
        
        # Extract tables
        tables_camelot = self.extract_tables_camelot()
        tables_pdfplumber = self.extract_tables_pdfplumber()
        all_tables = tables_camelot + tables_pdfplumber
        
        # Combine text content with better structure
        combined_text = ""
        for page_data in pages_data:
            combined_text += f"\n{'='*50}\n"
            combined_text += f"PAGE {page_data['page_num']}\n"
            combined_text += f"{'='*50}\n"
            
            # Clean and structure the text better
            page_text = page_data['text']
            # Remove excessive whitespace but preserve paragraph structure
            cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', page_text)
            cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
            
            combined_text += cleaned_text + "\n"
        
        # Add tables with better formatting
        for i, table in enumerate(all_tables, 1):
            combined_text += f"\n{'='*50}\n"
            combined_text += f"TABLE {i}\n"
            combined_text += f"{'='*50}\n"
            
            # Clean table data and format nicely
            table_clean = table.fillna('').astype(str)
            
            # Create a more readable table format
            combined_text += "Table Data:\n"
            combined_text += table_clean.to_string(index=False, max_cols=None, max_rows=None)
            combined_text += "\n\nTable Summary:\n"
            combined_text += f"Rows: {len(table_clean)}, Columns: {len(table_clean.columns)}\n"
            combined_text += f"Column Names: {', '.join(table_clean.columns.tolist())}\n\n"
        
        logger.info(f"Combined content: {len(combined_text)} chars, {len(pages_data)} pages, {len(all_tables)} tables")
        return combined_text, pages_data, all_tables


class RAGChatbot:
    """
    Enhanced RAG Chatbot with improved response quality and formatting
    """
    
    def __init__(self, groq_api_key: str = None):
        # Use environment variables if keys not provided
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        logger.info("Initializing RAG Chatbot")
        logger.debug(f"Groq API key present: {bool(self.groq_api_key)}")
        
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
    def initialize_embeddings(self, retry_count: int = 3, retry_delay: int = 5):
        """Initialize BAAI embeddings with retry logic"""
        logger.info("Initializing BAAI embeddings")
        
        for attempt in range(retry_count):
            try:
                self.embeddings = HuggingFaceBgeEmbeddings(
                    model_name="BAAI/bge-base-en-v1.5",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Test the embeddings
                test_embedding = self.embeddings.embed_query("test")
                logger.info(f"BAAI embeddings initialized successfully (dim: {len(test_embedding)})")
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if "DNS" in str(e) or "Timeout" in str(e):
                    logger.info(f"Network issue detected. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                elif attempt == retry_count - 1:
                    logger.error("Failed to initialize embeddings after all retries", exc_info=True)
                    raise
                    
        return False
        
    def initialize_llm(self):
        """Initialize Groq LLM with enhanced settings"""
        logger.info("Initializing Groq LLM with gemma2-9b-it model")
        try:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="gemma2-9b-it",
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),  # Lower temperature for more focused responses
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "3072")),     # Increased for better responses
                request_timeout=120
            )
            logger.info("Groq LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
            raise
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        logger.info(f"Creating FAISS vector store with {len(documents)} documents")
        
        try:
            # Create vector store with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore = FAISS.from_documents(
                        documents=documents,
                        embedding=self.embeddings
                    )
                    logger.info("FAISS vector store created successfully")
                    return vectorstore
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                        time.sleep(5)
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}", exc_info=True)
            raise
    
    def split_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
        """Split text into chunks for embedding with improved chunking strategy"""
        chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1200"))
        chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "300"))
        
        logger.info(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n" + "="*50 + "\n",  # Page/section separators
                "\n\n",                # Paragraph breaks
                "\n",                  # Line breaks
                ". ",                  # Sentence endings
                "! ",                  # Exclamation endings
                "? ",                  # Question endings
                "; ",                  # Semicolons
                ", ",                  # Commas
                " ",                   # Spaces
                ""                     # Characters
            ]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Add metadata to each chunk for better retrieval
        documents = []
        for i, chunk in enumerate(chunks):
            # Try to identify if chunk contains table data
            is_table = "TABLE" in chunk and ("Rows:" in chunk or "Columns:" in chunk)
            
            # Try to identify page information
            page_match = re.search(r'PAGE (\d+)', chunk)
            page_num = page_match.group(1) if page_match else "unknown"
            
            metadata = {
                "source": "pdf",
                "chunk_id": i,
                "page": page_num,
                "is_table": is_table,
                "char_count": len(chunk)
            }
            
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        logger.info(f"Text split into {len(documents)} chunks")
        return documents
    
    def process_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """Process PDF and create vector store"""
        logger.info(f"Starting PDF processing: {pdf_path}")
        
        try:
            # Extract content from PDF
            processor = PDFProcessor(pdf_path)
            combined_text, pages_data, tables = processor.get_combined_content()
            
            # Split text into chunks
            documents = self.split_text(combined_text)
            
            # Create vector store
            logger.info("Creating embeddings and vector store...")
            self.vectorstore = self.create_vector_store(documents)
            
            # Save metadata
            metadata = {
                "pdf_name": Path(pdf_path).name,
                "pages": len(pages_data),
                "tables": len(tables),
                "chunks": len(documents),
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"PDF processing completed: {metadata}")
            return True, f"Successfully processed PDF with {len(pages_data)} pages and {len(tables)} tables"
            
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def setup_qa_chain(self):
        """Setup conversational QA chain with improved prompt"""
        logger.info("Setting up QA chain")
        
        try:
            # Enhanced prompt template for better responses
            prompt_template = """You are an expert document analyst providing comprehensive and well-structured answers based on provided source material.

INSTRUCTIONS:
1. Answer ONLY using the information from the provided sources below
2. Structure your response in clear, readable markdown format
3. Provide detailed explanations with proper organization
4. Use bullet points, numbered lists, and headers when appropriate
5. If you find relevant information, synthesize it into a coherent response
6. If the sources don't contain sufficient information, clearly state: "Based on the provided documents, I cannot find complete information to answer this question."

QUESTION: {question}

SOURCE DOCUMENTS:
{context}

RESPONSE REQUIREMENTS:
- Use markdown formatting (##, ###, -, *, etc.)
- Organize information logically with headers and subheaders
- Provide specific details when available
- Include relevant data, numbers, or examples from the sources
- Summarize key points when dealing with lengthy information
- Maintain accuracy and avoid speculation

Your detailed response:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Enhanced retriever with better search parameters
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": int(os.getenv("RETRIEVER_K", "6")),  # Retrieve more chunks
                    "fetch_k": 20,  # Fetch more candidates before filtering
                }
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                verbose=True  # Enable verbose mode for better debugging
            )
            
            logger.info("Enhanced QA chain setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {str(e)}", exc_info=True)
            raise
    
    def post_process_response(self, response: str) -> str:
        """Post-process the LLM response for better formatting"""
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        
        # Ensure proper markdown headers
        response = re.sub(r'^([A-Z][^.\n]*):(?=\s)', r'## \1', response, flags=re.MULTILINE)
        
        # Fix bullet points
        response = re.sub(r'^[\s]*[-*]\s*', '- ', response, flags=re.MULTILINE)
        
        # Fix numbered lists
        response = re.sub(r'^[\s]*(\d+)[\.\)]\s*', r'\1. ', response, flags=re.MULTILINE)
        
        return response.strip()
    
    def ask(self, question: str) -> Dict:
        """Ask a question to the chatbot with enhanced response processing"""
        logger.info(f"Processing question: {question[:100]}...")
        
        if not self.qa_chain:
            logger.warning("QA chain not initialized")
            return {
                "answer": "Please upload and process a PDF first.",
                "sources": []
            }
        
        try:
            start_time = time.time()
            
            # Enhance the question for better retrieval
            enhanced_question = self.enhance_question(question)
            logger.debug(f"Enhanced question: {enhanced_question}")
            
            result = self.qa_chain({"question": enhanced_question})
            elapsed_time = time.time() - start_time
            
            # Post-process the response
            formatted_answer = self.post_process_response(result["answer"])
            
            logger.info(f"Question answered in {elapsed_time:.2f} seconds")
            
            # Extract and format source texts with metadata
            sources = []
            if "source_documents" in result:
                for i, doc in enumerate(result["source_documents"], 1):
                    source_info = {
                        "content": doc.page_content[:300] + "...",
                        "metadata": doc.metadata,
                        "relevance_score": f"Source {i}"
                    }
                    sources.append(source_info)
                logger.debug(f"Retrieved {len(sources)} source documents")
            
            return {
                "answer": formatted_answer,
                "sources": sources
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "answer": f"**Error:** {error_msg}",
                "sources": []
            }
    
    def enhance_question(self, question: str) -> str:
        """Enhance the user question for better retrieval"""
        # Add context keywords that might help retrieval
        enhanced = question
        
        # If question is very short, expand it slightly
        if len(question.split()) <= 3:
            enhanced = f"Please provide detailed information about: {question}"
        
        return enhanced
    
    def clear_memory(self):
        """Clear conversation memory"""
        logger.info("Clearing conversation memory")
        self.memory.clear()
    
    def save_vectorstore(self, path: str):
        """Save vector store to disk"""
        if self.vectorstore:
            logger.info(f"Saving vector store to: {path}")
            self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str):
        """Load vector store from disk"""
        logger.info(f"Loading vector store from: {path}")
        self.vectorstore = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )


def create_env_template():
    """Create a template .env file if it doesn't exist"""
    env_path = Path(".env")
    if not env_path.exists():
        template = """# API Keys
GROQ_API_KEY=your_groq_api_key_here

# Model Settings
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=3072

# Chunking Settings
CHUNK_SIZE=1200
CHUNK_OVERLAP=300

# Retriever Settings
RETRIEVER_K=6

# Logging
LOG_LEVEL=INFO
"""
        with open(env_path, 'w') as f:
            f.write(template)
        logger.info("Created .env template file")


def main():
    st.set_page_config(
        page_title="Enhanced PDF RAG Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    # Create .env template if needed
    create_env_template()
    
    st.title("📚 Enhanced PDF RAG Chatbot")
    st.markdown("Upload a PDF and get well-formatted, comprehensive answers using AI-powered search")
    
    # Display logger status in sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show logging status
        with st.expander("📊 System Status", expanded=False):
            st.success("✅ Enhanced Logging Active")
            st.text(f"Log Level: {logger.level}")
            st.text(f"Log File: logs/rag_chatbot_{datetime.now().strftime('%Y%m%d')}.log")
            
            # Check environment variables
            env_status = {
                "GROQ_API_KEY": "✅" if os.getenv("GROQ_API_KEY") else "❌",
            }
            st.text("Environment Variables:")
            for key, status in env_status.items():
                st.text(f"  {status} {key}")
        
        # API Keys (can override env vars)
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            placeholder="Enter your Groq API key",
            help="Get your API key from Groq Console"
        )
        
        # Enhanced settings with better defaults
        with st.expander("Advanced Settings"):
            chunk_size = st.slider(
                "Chunk Size", 
                800, 2000, 
                int(os.getenv("CHUNK_SIZE", "1200"))
            )
            chunk_overlap = st.slider(
                "Chunk Overlap", 
                100, 500, 
                int(os.getenv("CHUNK_OVERLAP", "300"))
            )
            temperature = st.slider(
                "Temperature", 
                0.0, 1.0, 
                float(os.getenv("LLM_TEMPERATURE", "0.1")),
                help="Lower values = more focused responses"
            )
            max_tokens = st.slider(
                "Max Tokens", 
                1024, 4096, 
                int(os.getenv("LLM_MAX_TOKENS", "3072")),
            )
            retriever_k = st.slider(
                "Documents Retrieved",
                3, 10,
                int(os.getenv("RETRIEVER_K", "6")),
                help="Number of relevant chunks to retrieve"
            )
        
        # File upload
        st.header("📁 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload the PDF you want to query"
        )
        
        process_button = st.button("🔄 Process PDF", type="primary", use_container_width=True)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.clear_memory()
            st.success("Chat history cleared!")
            logger.info("User cleared chat history")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Process PDF
    if process_button and uploaded_file and groq_api_key:
        with st.spinner("Processing PDF with enhanced chunking... This may take a few moments."):
            try:
                logger.info(f"User initiated enhanced PDF processing: {uploaded_file.name}")
                
                # Save uploaded file temporarily
                temp_path = Path("temp_pdf.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize chatbot
                chatbot = RAGChatbot(groq_api_key)
                
                # Initialize embeddings with retry
                with st.status("Initializing enhanced components...", expanded=True) as status:
                    st.write("🔧 Initializing BAAI Embeddings...")
                    chatbot.initialize_embeddings()
                    st.write("✅ Embeddings ready")
                    
                    st.write("🤖 Initializing Enhanced LLM...")
                    chatbot.initialize_llm()
                    st.write("✅ LLM ready")
                    
                    status.update(label="📄 Processing PDF with enhanced chunking...", state="running")
                
                # Process PDF
                success, message = chatbot.process_pdf(str(temp_path))
                
                if success:
                    chatbot.setup_qa_chain()
                    st.session_state.chatbot = chatbot
                    st.session_state.pdf_processed = True
                    st.success(f"✅ {message}")
                    logger.info(f"Enhanced PDF processed successfully: {uploaded_file.name}")
                    
                    # Clean up temp file
                    temp_path.unlink()
                else:
                    st.error(f"❌ {message}")
                    logger.error(f"Enhanced PDF processing failed: {message}")
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Enhanced Chat")
        
        if not st.session_state.pdf_processed:
            st.info("👈 Please upload and process a PDF to start chatting with enhanced responses")
        else:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        # Render markdown for assistant responses
                        st.markdown(message["content"])
                    else:
                        st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a detailed question about your PDF..."):
                logger.info(f"User question: {prompt}")
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document and generating comprehensive response..."):
                        response = st.session_state.chatbot.ask(prompt)
                        
                        # Display the formatted response
                        st.markdown(response["answer"])
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"]
                        })
                        
                        logger.info(f"Enhanced assistant response generated: {len(response['answer'])} chars")
                        
                        # Show enhanced sources if available
                        # Show enhanced sources if available
                        if response["sources"]:
                            with st.expander("📎 View Source Details", expanded=False):
                                for idx, source in enumerate(response["sources"]):
                                    # Defensive check: ensure source is a dict
                                    if isinstance(source, dict):
                                        st.markdown(f"**{source.get('relevance_score', f'Source {idx+1}')}**")
                                        st.text(source.get("content", "")[:500] + "...")
                                        
                                        metadata = source.get("metadata")
                                        if metadata:
                                            st.json(metadata)
                                    else:
                                        # Log or show error if source is malformed
                                        st.warning(f"Unexpected source format (type: {type(source)}): {str(source)[:100]}")
                                    
                                    st.divider()
    
    with col2:
        st.header("📊 Enhanced Document Info")
        
        if st.session_state.pdf_processed:
            st.success("✅ PDF Processed with Enhanced Chunking")
            
            # Show document stats
            if uploaded_file:
                st.metric("File Name", uploaded_file.name)
                st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
            
            # Enhanced features info
            with st.expander("🔧 Enhanced Features"):
                st.write("✅ **Improved Text Chunking**")
                st.write("- Better preservation of document structure")
                st.write("- Enhanced table detection and formatting")
                st.write("- Metadata-rich chunks for better retrieval")
                
                st.write("✅ **Advanced Response Generation**")
                st.write("- Structured markdown formatting")
                st.write("- Comprehensive answer synthesis")
                st.write("- Lower temperature for focused responses")
                
                st.write("✅ **Enhanced Retrieval**")
                st.write("- Increased chunk retrieval (6 vs 4)")
                st.write("- Better similarity search parameters")
                st.write("- Source metadata preservation")
            
            # Response quality tips
            with st.expander("💡 Tips for Better Responses"):
                st.write("**For best results, ask questions that are:**")
                st.write("- Specific and detailed")
                st.write("- Related to document content")
                st.write("- Clear about what information you need")
                
                st.write("**Example good questions:**")
                st.code("""
• "What are the main findings in section 3?"
• "Summarize the financial data in table format"
• "Explain the methodology used in this study"
• "What recommendations are provided?"
                """)
            
            # View logs
            if st.button("📜 View Recent Logs", use_container_width=True):
                log_file = Path("logs") / f"rag_chatbot_{datetime.now().strftime('%Y%m%d')}.log"
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        logs = f.readlines()[-50:]  # Last 50 lines
                    with st.expander("Recent Logs", expanded=True):
                        for log_line in logs:
                            if "ERROR" in log_line:
                                st.error(log_line.strip())
                            elif "WARNING" in log_line:
                                st.warning(log_line.strip())
                            elif "INFO" in log_line:
                                st.info(log_line.strip())
                            else:
                                st.text(log_line.strip())
            
            # Export chat history
            if st.session_state.messages:
                if st.button("💾 Export Chat History", use_container_width=True):
                    # Create formatted export
                    export_data = {
                        "export_timestamp": datetime.now().isoformat(),
                        "pdf_name": uploaded_file.name if uploaded_file else "Unknown",
                        "total_messages": len(st.session_state.messages),
                        "conversation": st.session_state.messages
                    }
                    
                    chat_json = json.dumps(export_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download Enhanced Chat History",
                        data=chat_json,
                        file_name=f"enhanced_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    logger.info("User exported enhanced chat history")
                    
                # Export as markdown
                if st.button("📝 Export as Markdown", use_container_width=True):
                    markdown_content = f"# Chat History - {uploaded_file.name if uploaded_file else 'PDF Document'}\n\n"
                    markdown_content += f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    markdown_content += "---\n\n"
                    
                    for i, message in enumerate(st.session_state.messages, 1):
                        role = "🧑 **User**" if message["role"] == "user" else "🤖 **Assistant**"
                        markdown_content += f"## Message {i} - {role}\n\n{message['content']}\n\n---\n\n"
                    
                    st.download_button(
                        label="📄 Download as Markdown",
                        data=markdown_content,
                        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
        else:
            st.info("No document loaded")
            
            # Show benefits of enhanced version
            with st.expander("🚀 Enhanced Features Preview", expanded=True):
                st.write("**This enhanced version provides:**")
                st.write("- 📝 **Markdown-formatted responses** with headers, lists, and structure")
                st.write("- 🎯 **Better chunk retrieval** with improved relevance scoring")
                st.write("- 📊 **Enhanced table processing** with better formatting")
                st.write("- 🔍 **More comprehensive answers** with detailed explanations")
                st.write("- 📈 **Improved response quality** through better prompting")
                st.write("- 🏷️ **Source metadata** showing page numbers and content types")
    
    # Footer with enhanced features
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔧 Enhanced Processing")
        st.write("- Improved text chunking")
        st.write("- Better table extraction")
        st.write("- Metadata preservation")
    
    with col2:
        st.markdown("### 🤖 Smarter Responses")
        st.write("- Structured markdown output")
        st.write("- Comprehensive synthesis")
        st.write("- Lower temperature for focus")
    
    with col3:
        st.markdown("### 📊 Better Retrieval")
        st.write("- Enhanced similarity search")
        st.write("- More relevant chunks")
        st.write("- Source tracking")


if __name__ == "__main__":
    # Add enhanced requirements info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Required Packages")
    st.sidebar.code("""
pip install streamlit
pip install langchain
pip install langchain-groq
pip install langchain-community
pip install faiss-cpu
pip install PyMuPDF
pip install pdfplumber
pip install camelot-py[cv]
pip install pandas
pip install numpy
pip install python-dotenv
pip install sentence-transformers
    """, language="bash")
    
    # Enhanced startup message
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Performance Tips")
    st.sidebar.write("- Use specific, detailed questions")
    st.sidebar.write("- Check the enhanced features panel")
    st.sidebar.write("- Review source metadata for context")
    st.sidebar.write("- Export conversations as markdown")
    
    # Log application start
    logger.info("="*50)
    logger.info("Enhanced Streamlit RAG Chatbot Application Started")
    logger.info("Enhanced features: Better chunking, markdown responses, improved retrieval")
    logger.info("="*50)
    
    main()