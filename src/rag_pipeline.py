import pandas as pd
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pathlib import Path
import os
import re

# Set working directory
os.chdir('C:/Users/daniel.shobe/Desktop/schoolify/B5W6/creditrust-complaint-analysis')
print(f"Working directory set to: {os.getcwd()}")

# Set up directories
DATA_DIR = Path('data')
VECTOR_STORE_DIR = Path('vector_store')

# Load metadata and FAISS index
chunked_df = pd.read_csv(VECTOR_STORE_DIR / 'chunk_metadata.csv')
index = faiss.read_index(str(VECTOR_STORE_DIR / 'complaint_index.faiss'))
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device set to use {model.device}")
generator = pipeline('text-generation', model='gpt2-medium', max_new_tokens=100)

# RAG pipeline function with debugged product filtering
def rag_pipeline(question, top_k=5):
    # Debug: Print product counts
    print("Available products in chunked_df:", chunked_df['Product'].unique())
    
    # Detect product in question
    product_keywords = {
        'BNPL': ['bnpl', 'buy now pay later', 'pay later'],
        'Credit Card': ['credit card', 'credit cards'],  # Handle singular and plural
        'Personal Loan': ['personal loan', 'payday loan', 'title loan', 'advance loan'],
        'Savings Account': ['savings account', 'bank account'],
        'Money Transfer': ['money transfer', 'virtual currency', 'money service']
    }
    target_product = None
    for product, keywords in product_keywords.items():
        if any(re.search(r'\b' + keyword + r'\b', question, re.IGNORECASE) for keyword in keywords):
            target_product = product
            break
    print(f"Detected product: {target_product}")
    
    # Filter chunks by product
    if target_product:
        valid_products = [target_product]
        if target_product == 'BNPL':
            valid_products.append('Payday loan, title loan, personal loan, or advance loan')
        elif target_product == 'Personal Loan':
            valid_products.append('Payday loan, title loan, personal loan, or advance loan')
        elif target_product == 'Savings Account':
            valid_products.append('Bank account or service')
        elif target_product == 'Money Transfer':
            valid_products.append('Money transfer, virtual currency, or money service')
        elif target_product == 'Credit Card':
            valid_products.append('Credit card')  # Map to singular in chunked_df
        filtered_df = chunked_df[chunked_df['Product'].isin(valid_products)]
        print(f"Filtered chunks: {len(filtered_df)}")
    else:
        filtered_df = chunked_df
        print("No product filter applied")
    
    # If no chunks after filtering
    if filtered_df.empty:
        return {
            'question': question,
            'answer': "No relevant complaints found for this product.",
            'retrieved_chunks': []
        }
    
    # Create a mapping of original indices to filtered indices
    filtered_indices = filtered_df.index.to_numpy()
    index_to_filtered = {i: idx for i, idx in enumerate(filtered_indices)}
    
    # Embed the question
    question_embedding = model.encode([question])[0]
    
    # Search FAISS index
    distances, indices = index.search(np.array([question_embedding], dtype=np.float32), top_k)
    
    # Filter retrieved indices
    retrieved_chunks = []
    for idx in indices[0]:
        if idx in filtered_indices:
            chunk_data = chunked_df.iloc[idx]
            retrieved_chunks.append({
                'Complaint ID': chunk_data['Complaint ID'],
                'Product': chunk_data['Product'],
                'chunk': chunk_data['chunk'],
                'distance': distances[0][list(indices[0]).index(idx)]
            })
    
    # If no valid chunks
    if not retrieved_chunks:
        return {
            'question': question,
            'answer': "No relevant complaints found for this product.",
            'retrieved_chunks': []
        }
    
    # Build context
    context = "\n".join([f"Complaint ID: {chunk['Complaint ID']}, Product: {chunk['Product']}, Text: {chunk['chunk']}" for chunk in retrieved_chunks])
    
    # Prompt template
    prompt = f"""You are a financial analyst assistant for CrediTrust. Answer questions about customer complaints using only the provided context. If the context lacks the answer, state that you don't have enough information. Be concise and evidence-based.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate response
    response = generator(prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
    answer = response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()
    
    return {
        'question': question,
        'answer': answer,
        'retrieved_chunks': retrieved_chunks
    }

# Evaluation questions
questions = [
    "Why are people unhappy with BNPL?",
    "What are common issues with Credit Cards?",
    "Are there fraud-related complaints for Money Transfers?",
    "What problems do customers face with Savings Accounts?",
    "Why do Personal Loan customers complain?",
    "Compare issues between BNPL and Credit Cards.",
    "What are the top reasons for complaints across all products?"
]

# Run evaluation
evaluation_results = []
for question in questions:
    result = rag_pipeline(question)
    # Placeholder scores
    score = 3
    comments = "Pending manual review of answer relevance and accuracy."
    evaluation_results.append({
        'Question': question,
        'Generated Answer': result['answer'],
        'Retrieved Sources': "\n".join([f"Complaint ID: {chunk['Complaint ID']}, Product: {chunk['Product']}, Text: {chunk['chunk'][:100]}..." for chunk in result['retrieved_chunks'][:2]]),
        'Quality Score': score,
        'Comments': comments
    })

# Save evaluation table
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv(DATA_DIR / 'rag_evaluation.csv', index=False)
print(f"Saved evaluation table to {DATA_DIR / 'rag_evaluation.csv'}")