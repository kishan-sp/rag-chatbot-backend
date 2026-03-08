import os
import fitz  # PyMuPDF
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # loads GROQ_API_KEY and other env vars from backend/.env

# Initialize FastAPI app
app = FastAPI(title="RAG Business Chatbot API")

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

@app.get("/")
def health_check():
    """Health check endpoint to verify backend is running."""
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = ""):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    # Create the /tmp directory locally if it doesn't exist to prevent Windows testing errors
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)

    # Save to disk
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # Extract text
    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file. It may be a scanned image PDF."
        )

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.create_documents([text])

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

if __name__ == "__main__":
    # Run the application using Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
