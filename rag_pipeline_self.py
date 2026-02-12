import json
import os
import pickle

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