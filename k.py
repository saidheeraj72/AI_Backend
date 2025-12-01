import streamlit as st
import os
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv
import json

# PDF Processing
import camelot
import PyPDF2
    
# Embeddings and Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# LLM
from groq import Groq

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./files")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
LOG_FILE = os.getenv("LOG_FILE", "./logs/chatbot.log")

# Setup logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

class RAGChatbot:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            logger.info(f"Extracted text from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_path):
        """Extract tables from PDF using Camelot"""
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            table_text = ""
            for i, table in enumerate(tables):
                table_text += f"\n\nTable {i+1}:\n"
                table_text += table.df.to_string()
            logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
            return table_text
        except Exception as e:
            logger.warning(f"Error extracting tables from {pdf_path}: {e}")
            return ""
    
    def process_pdfs(self, pdf_directory):
        """Process all PDFs in directory"""
        documents = []
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return documents
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            # Extract text
            text = self.extract_text_from_pdf(str(pdf_path))
            
            # Extract tables
            tables = self.extract_tables_from_pdf(str(pdf_path))
            
            # Combine text and tables
            full_content = text + tables
            
            if full_content.strip():
                doc = Document(
                    page_content=full_content,
                    metadata={"source": pdf_path.name}
                )
                documents.append(doc)
        
        return documents
    
    def create_vectorstore(self, pdf_directory):
        """Create FAISS vectorstore from PDFs"""
        logger.info("Starting vectorstore creation")
        
        # Process PDFs
        documents = self.process_pdfs(pdf_directory)
        
        if not documents:
            logger.error("No documents to process")
            return False
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} text chunks")
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(splits, self.embedding_model)
        
        # Save vectorstore
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        self.vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info(f"Vectorstore saved to {FAISS_INDEX_PATH}")
        
        return True
    
    def load_vectorstore(self):
        """Load existing FAISS vectorstore"""
        try:
            self.vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Vectorstore loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            return False
    
    def retrieve_context(self, query, k=4):
        """Retrieve relevant context from vectorstore"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_response(self, query, context_docs):
        """Generate response using Groq API"""
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Use the following context to answer the question. If you cannot find the answer in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Call Groq API
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gemma2-9b-it",
                temperature=0.7,
                max_tokens=1024,
            )
            
            response = chat_completion.choices[0].message.content
            logger.info(f"Generated response for query: {query[:50]}...")
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def chat(self, query):
        """Main chat function"""
        logger.info(f"User query: {query}")
        
        # Retrieve context
        context_docs = self.retrieve_context(query)
        
        if not context_docs:
            response = "I couldn't find relevant information in the documents."
            logger.warning("No relevant documents found")
            return response, []
        
        # Generate response
        response = self.generate_response(query, context_docs)
        
        # Log interaction
        self.log_interaction(query, response, context_docs)
        
        return response, context_docs
    
    def log_interaction(self, query, response, context_docs):
        """Log chat interaction"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "sources": [doc.metadata.get("source", "unknown") for doc in context_docs]
        }
        
        log_path = "./logs/interactions.jsonl"
        os.makedirs("./logs", exist_ok=True)
        
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    
    st.title("📚 RAG Chatbot with PDF Processing")
    st.markdown("---")
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        st.session_state.messages = []
        st.session_state.vectorstore_ready = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write(f"**PDF Directory:** {PDF_DIRECTORY}")
        st.write(f"**FAISS Index:** {FAISS_INDEX_PATH}")
        st.write(f"**Embedding Model:** BAAI/bge-base-en-v1.5")
        st.write(f"**LLM:** gemma2-9b-it (Groq)")
        
        st.markdown("---")
        
        # Process PDFs button
        if st.button("🔄 Process PDFs", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                if st.session_state.chatbot.create_vectorstore(PDF_DIRECTORY):
                    st.session_state.vectorstore_ready = True
                    st.success("✅ PDFs processed successfully!")
                else:
                    st.error("❌ Error processing PDFs")
        
        # Load existing vectorstore
        if st.button("📂 Load Existing Index", use_container_width=True):
            with st.spinner("Loading vectorstore..."):
                if st.session_state.chatbot.load_vectorstore():
                    st.session_state.vectorstore_ready = True
                    st.success("✅ Vectorstore loaded!")
                else:
                    st.error("❌ Error loading vectorstore")
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Status
        st.markdown("---")
        st.subheader("📊 Status")
        if st.session_state.vectorstore_ready:
            st.success("🟢 Vectorstore Ready")
        else:
            st.warning("🟡 Vectorstore Not Loaded")
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.vectorstore_ready:
                st.error("Please process PDFs or load an existing index first!")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response, sources = st.session_state.chatbot.chat(prompt)
                        st.write(response)
                        
                        # Show sources
                        if sources:
                            with st.expander("📄 Sources"):
                                for i, doc in enumerate(sources, 1):
                                    st.write(f"**Source {i}:** {doc.metadata.get('source', 'unknown')}")
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("📈 Logs")
        
        # Show recent logs
        try:
            if os.path.exists("./logs/interactions.jsonl"):
                with open("./logs/interactions.jsonl", "r") as f:
                    lines = f.readlines()
                    recent_logs = lines[-5:]  # Last 5 interactions
                    
                    for line in reversed(recent_logs):
                        log = json.loads(line)
                        with st.expander(f"🕐 {log['timestamp'][:19]}"):
                            st.write(f"**Q:** {log['query'][:100]}...")
                            st.write(f"**A:** {log['response'][:100]}...")
                            st.write(f"**Sources:** {', '.join(log['sources'])}")
        except Exception as e:
            st.write("No logs yet")

if __name__ == "__main__":
    main()