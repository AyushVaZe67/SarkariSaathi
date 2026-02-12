import json
import os
import pickle

with open('data/girls_education_maharashtra.json', 'r') as f:
    data = json.load(f)

print(data[0])