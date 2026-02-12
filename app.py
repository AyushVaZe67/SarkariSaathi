from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

app = FastAPI(title="SarkariSaathi - Government Schemes Chatbot")

# ----------------------------
# Initialize RAG Components
# ----------------------------
print("🔄 Loading data and models...")

# Load data
with open("data/girls_education_maharashtra.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
important_fields = [
    "description",
    "benefits",
    "eligibility",
    "application_process",
    "documents_required",
    "tenure",
    "relaxation_priority",
    "exclusions",
    "faqs"
]

for item in data:
    scheme_name = item.get("scheme_name", "")
    scheme_id = item.get("scheme_id", "")
    state = item.get("state", "")
    ministry = item.get("ministry", "")

    for field in important_fields:
        content = item.get(field, "")
        if content:
            chunk = f"""
            Scheme Name: {scheme_name}
            Scheme ID: {scheme_id}
            State: {state}
            Ministry: {ministry}
            Section: {field.upper()}

            {content}
            """
            documents.append(chunk.strip())

print(f"📄 Loaded {len(documents)} document chunks")

# Create embeddings
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embed_model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ FAISS index ready")

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=groq_api_key)

print("✅ Groq client initialized")
print("🚀 Server ready!\n")


# ----------------------------
# Pydantic Models
# ----------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []


class ChatResponse(BaseModel):
    answer: str
    success: bool = True


# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat requests with RAG
    """
    try:
        query = request.query
        chat_history = [msg.dict() for msg in request.history]

        print(f"\n📨 Received query: {query}")
        print(f"📚 History length: {len(chat_history)}")

        MAX_HISTORY = 6

        # Build conversation context for better retrieval
        previous_user_msgs = [
            msg["content"] for msg in chat_history if msg["role"] == "user"
        ]

        conversation_context = " ".join(previous_user_msgs + [query])

        # Embed conversation-aware query
        query_embedding = embed_model.encode([conversation_context])
        query_embedding = np.array(query_embedding).astype("float32")

        # Retrieve top chunks
        D, I = index.search(query_embedding, k=8)
        retrieved_docs = [documents[i] for i in I[0]]
        context = "\n\n".join(retrieved_docs)

        print(f"🔍 Retrieved {len(retrieved_docs)} relevant documents")

        # Add current user query to memory
        chat_history.append({"role": "user", "content": query})

        # Keep memory limited
        chat_history = chat_history[-MAX_HISTORY:]

        # Build messages for LLM
        messages = [
            {
                "role": "system",
                "content": "You are SarkariSaathi, an expert AI assistant for government schemes. Use only the provided context."
            }
        ]

        # Add chat memory
        messages.extend(chat_history)

        # Add RAG context as final instruction
        messages.append({
            "role": "user",
            "content": f"""
Use ONLY the information provided in the context below.
Do NOT add information not present in the context.
If answer is missing, say:
"Information not available in the provided data."

---------------------
CONTEXT:
{context}
---------------------

Answer clearly and concisely.
"""
        })

        print(f"🤖 Calling Groq API...")

        # Call Groq LLM
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        print(f"✅ Response generated successfully")

        return ChatResponse(answer=answer, success=True)

    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_loaded": len(documents),
        "model": "llama-3.1-8b-instant"
    }


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    print("=" * 60)
    print("🌟 SarkariSaathi Server Starting...")
    print("=" * 60)
    print(f"📍 Access the app at: http://127.0.0.1:8000")
    print(f"📍 Or use: http://localhost:8000")
    print("=" * 60)
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")