# creditrust-complaint-analysis
RAG-powered chatbot to analyze customer complaints for CrediTrust Financial.
# CrediTrust Complaint Analysis

This repository contains the implementation of an intelligent complaint-answering chatbot for CrediTrust Financial, built using Retrieval-Augmented Generation (RAG). The chatbot analyzes customer complaints from the Consumer Financial Protection Bureau (CFPB) dataset to provide actionable insights for internal stakeholders across Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers.

## Project Structure
- `data/`: Raw and processed datasets, plus EDA visualizations.
- `notebooks/`: Jupyter Notebooks for EDA and preprocessing.
- `src/`: Python scripts for data preprocessing, RAG pipeline, and UI.
- `vector_store/`: Persisted vector database (FAISS/ChromaDB).
- `reports/`: Interim and final reports, including UI screenshots.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Excludes unnecessary files.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/moheranus/creditrust-complaint-analysis.git
   cd creditrust-complaint-analysis