# ===============================
# SarkariSaathi – RAG Pipeline
# JSON-based Government Schemes
# ===============================

from dotenv import load_dotenv
import os
import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ===============================
# CONFIGURATION
# ===============================

SCHEME_JSON_PATH = "data/schemes.json"   # <-- your JSON file
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TOP_K = 3


# ===============================
# ENVIRONMENT
# ===============================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found in .env file")


# ===============================
# LOAD JSON → DOCUMENTS
# ===============================

def load_scheme_documents(json_path: str):
    """
    Converts each scheme JSON object into a LangChain Document
    """
    with open(json_path, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    documents = []

    for scheme in schemes:
        content = f"""
Scheme Name: {scheme.get('scheme_name')}
Scheme ID: {scheme.get('scheme_id')}
State: {scheme.get('state')}
Department: {scheme.get('department')}
Launch Year: {scheme.get('launch_year', 'N/A')}

Target Group: {', '.join(scheme.get('target_group', []))}
Education Level: {', '.join(scheme.get('education_level', []))}

Eligibility Details:
{json.dumps(scheme.get('eligibility', {}), indent=2)}

Benefits:
{json.dumps(scheme.get('benefits', {}), indent=2)}

Application Process:
{json.dumps(scheme.get('application_process', {}), indent=2)}

Documents Required:
{', '.join(scheme.get('documents_required', []))}

Special Facilities:
{', '.join(scheme.get('special_facilities', []))}

Notes:
{', '.join(scheme.get('notes', []))}

Official Source:
{json.dumps(scheme.get('source', {}), indent=2)}
"""

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "scheme_id": scheme.get("scheme_id"),
                    "state": scheme.get("state"),
                    "scheme_type": scheme.get("source", {}).get("scheme_type")
                }
            )
        )

    return documents


# ===============================
# TEXT SPLITTING
# ===============================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    return splitter.split_documents(documents)


# ===============================
# VECTOR STORE
# ===============================

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_db


# ===============================
# LOAD LLM (GROQ)
# ===============================

def load_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY
    )


# ===============================
# RAG ANSWER FUNCTION
# ===============================

def answer_query(query, vector_db, llm):
    retriever = vector_db.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = ChatPromptTemplate.from_template("""
You are SarkariSaathi, an AI assistant that recommends Indian government schemes.

Use ONLY the information from the context below.
If no suitable scheme matches, say clearly:
"No suitable government scheme found for your criteria."

Context:
{context}

User Question:
{question}

Answer in clear, simple, structured points.
""")

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    return response.content


# ===============================
# MAIN
# ===============================

def main():
    print("🚀 Loading schemes...")
    documents = load_scheme_documents(SCHEME_JSON_PATH)

    print("✂️ Splitting documents...")
    chunks = split_documents(documents)

    print("📦 Building FAISS vector store...")
    vector_db = build_vector_store(chunks)

    print("🤖 Loading Groq LLM...")
    llm = load_llm()

    print("\n✅ SarkariSaathi RAG is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("👤 Ask your question: ")

        if query.lower() == "exit":
            break

        answer = answer_query(query, vector_db, llm)
        print("\n🧠 Answer:\n")
        print(answer)
        print("-" * 70)


if __name__ == "__main__":
    main()
