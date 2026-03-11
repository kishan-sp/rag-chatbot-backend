import os
import tempfile
import fitz  # PyMuPDF
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.schema import Document
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()  # loads GROQ_API_KEY and other env vars from backend/.env

# Initialize FastAPI app
app = FastAPI(title="RAG Business Chatbot API")

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global sessions dictionary
# Keyed by session_id, stores FAISS vector store
sessions: dict[str, FAISS] = {}

class ChatRequest(BaseModel):
    question: str = Field(..., max_length=2000, description="The user's question, limited to 2000 characters.")
    session_id: str
    chat_history: List[Dict[str, Any]] = []

class SourceSnippet(BaseModel):
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]

@app.get("/")
def health_check():
    """Health check endpoint to verify backend is running."""
    return {"status": "ok"}

@app.post("/upload")
@limiter.limit("20/day")
async def upload(request: Request, file: UploadFile = File(...), session_id: str = ""):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    tmp_path = os.path.join(tempfile.gettempdir(), file.filename)

    # Save to disk
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # Extract text
    doc = fitz.open(tmp_path)
    docs_with_metadata = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text.strip():
            docs_with_metadata.append({
                "text": page_text,
                "metadata": {"page": page_num + 1}
            })
    doc.close()

    if not docs_with_metadata:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file. It may be a scanned image PDF."
        )

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    documents = [
        Document(page_content=d["text"], metadata=d["metadata"])
        for d in docs_with_metadata
    ]
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Document could not be split into chunks. File may be empty."
        )

    # Embed + store in FAISS (Using local fallback embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        sessions[session_id] = vector_store
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    print(f"DEBUG: Active Sessions = {list(sessions.keys())}")

    return {
        "status": "ready",
        "chunk_count": len(chunks),
        "session_id": session_id
    }

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/day")
async def chat(request: Request, chat_request: ChatRequest):
    if not chat_request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if chat_request.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")
        
    # Get the vector store for this session
    vector_store = sessions[chat_request.session_id]
    
    # Initialize the retriever (scoped to this request)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Initialize the LLM (Groq)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    
    # Initialize memory and populate with chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    for msg in chat_request.chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            memory.chat_memory.add_user_message(content)
        elif role == "assistant":
            memory.chat_memory.add_ai_message(content)
            
    # Build the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    
    # Invoke the chain
    result = chain.invoke({"question": chat_request.question})
    
    # Process source documents
    sources_list = []
    seen_pages = set()
    
    for doc in result.get("source_documents", []):
        page = doc.metadata.get("page", 0)
        # Deduplicate sources by page
        if page not in seen_pages:
            seen_pages.add(page)
            # Create a short snippet (first 200 characters)
            snippet = doc.page_content[:200]
            sources_list.append(SourceSnippet(page=page, snippet=snippet))
    
    return ChatResponse(
        answer=result.get("answer", ""),
        sources=sources_list
    )

if __name__ == "__main__":
    # Run the application using Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
