import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

with open('data/girls_education_maharashtra.json', 'r') as f:
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

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)
embeddings = np.array(embeddings)

# print(embeddings[0])
# print(len(embeddings))

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Total schemes indexed:", index.ntotal)

def retrieve(query, k=3):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector)

    distances, indices = index.search(query_vector, k)

    results = [documents[i] for i in indices[0]]
    return results