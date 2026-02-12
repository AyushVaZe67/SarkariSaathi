import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------------
# 1. Load Data
# ----------------------------
with open("data/girls_education_maharashtra.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data[0])

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

print(len(documents))
print(documents[0])

# ----------------------------
# 2. Create Embeddings
# ----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = embed_model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

# print(embeddings[0])
# print(len(embeddings))

# ----------------------------
# 3. Create FAISS Index
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ Index ready")

# ----------------------------
# 4. Groq Client
# ----------------------------

# Get API key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# ----------------------------
# 5. Chat Loop
# ----------------------------
chat_history = []
MAX_HISTORY = 6  # keep last 6 messages only

while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    # ----------------------------
    # 1. Build conversation context for better retrieval
    # ----------------------------
    previous_user_msgs = [
        msg["content"] for msg in chat_history if msg["role"] == "user"
    ]

    conversation_context = " ".join(previous_user_msgs + [query])

    # Embed conversation-aware query
    query_embedding = embed_model.encode([conversation_context])
    query_embedding = np.array(query_embedding).astype("float32")

    # ----------------------------
    # 2. Retrieve Top Chunks
    # ----------------------------
    D, I = index.search(query_embedding, k=8)

    retrieved_docs = [documents[i] for i in I[0]]
    context = "\n\n".join(retrieved_docs)

    # ----------------------------
    # 3. Add current user query to memory
    # ----------------------------
    chat_history.append({"role": "user", "content": query})

    # Keep memory limited
    chat_history = chat_history[-MAX_HISTORY:]

    # ----------------------------
    # 4. Build Messages for LLM
    # ----------------------------
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

    # ----------------------------
    # 5. Call Groq LLM
    # ----------------------------
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    print("\nBot:", answer)

    # ----------------------------
    # 6. Save assistant reply to memory
    # ----------------------------
    chat_history.append({"role": "assistant", "content": answer})
