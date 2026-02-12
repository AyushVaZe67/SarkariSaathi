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

for item in data:
    text = f"""
    Scheme ID: {item.get("scheme_id", "")} 
    Scheme Name: {item.get("scheme_name", "")}
    State: {item.get("state", "")}
    Ministry: {item.get("ministry", "")}
    Description: {item.get("description", "")}
    Benefits: {item.get("benefits", "")}
    Eligibility: {item.get("eligibility", "")}
    Relaxation / Priority: {item.get("relaxation_priority", "")}
    Exclusions: {item.get("exclusions", "")}
    Tenure: {item.get("tenure", "")}
    Application Process: {item.get("application_process", "")}
    Documents Required: {item.get("documents_required", "")}
    FAQs: {item.get("faqs", "")}
    Source URL: {item.get("source_url", "")}
    """

    documents.append(text.strip())

# print(len(documents))

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
while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    # Embed query
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search top 3
    D, I = index.search(query_embedding, k=3)

    retrieved_docs = [documents[i] for i in I[0]]
    context = "\n\n".join(retrieved_docs)

    # Create prompt
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}

    Answer clearly:
    """

    # Call Groq LLM
    response = client.chat.completions.create(
        model="llama3-8b-8192",   # or your preferred Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    print("\nBot:", answer)